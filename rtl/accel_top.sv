// Milestone 2: tiled weight-stationary accelerator.
//
// Supports arbitrary M x N x K GEMM on a fixed N_ARR x N_ARR array via
// tiling, including ragged tiles handled at their true dimensions with
// NO zero-padding, and multi-K-tile accumulation via C read-modify-write.
//
// Still out of scope (by design): output-stationary dataflow, the
// adaptive policy, double buffering, and hardware performance counters.
//
// Loop structure (weight-stationary):
//
//   for n_tile:                     // output column block
//     for k_tile:                   // reduction block
//       fetch B tile -> weight load into the array
//       for m_chunk:                // streamed activations
//         fetch A chunk -> compute -> writeback
//
// The m loop is innermost specifically so that weights stay RESIDENT
// across every M chunk -- reloading them per chunk would waste k_tile
// cycles per chunk and defeat the point of weight-stationary execution.
//
// Width discipline: counters are compared against localparams cast to
// the counter's own width, so the file is clean under -Wall with no
// WIDTHEXPAND suppression.
module accel_top #(
    parameter int N_ARR        = 4,
    parameter int DATA_W       = 8,
    parameter int ACC_W        = 32,
    parameter int ADDR_W       = 16,
    parameter int STREAM_DEPTH = 16,   // max M rows resident per chunk
    parameter int MAX_K        = 256
) (
    input  logic clk,
    // rst_n is used both as an async FF reset and inside assertion
    // `disable iff` clauses -- the standard, safe pattern for this reset
    // style, and the only reason SYNCASYNCNET is waived.
    /* verilator lint_off SYNCASYNCNET */
    input  logic rst_n,
    /* verilator lint_on SYNCASYNCNET */

    input  logic start,
    output logic busy,
    output logic done,
    output logic error,

    input  logic [15:0] cfg_m,
    input  logic [15:0] cfg_n,
    input  logic [15:0] cfg_k,
    input  logic [1:0]  cfg_policy,

    output logic [ADDR_W-1:0]        mem_a_addr,
    output logic                     mem_a_rd_en,
    input  logic signed [DATA_W-1:0] mem_a_rdata,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic                     mem_a_rvalid,
    /* verilator lint_on UNUSEDSIGNAL */

    output logic [ADDR_W-1:0]        mem_b_addr,
    output logic                     mem_b_rd_en,
    input  logic signed [DATA_W-1:0] mem_b_rdata,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic                     mem_b_rvalid,
    /* verilator lint_on UNUSEDSIGNAL */

    output logic [ADDR_W-1:0]       mem_c_addr,
    output logic                    mem_c_rd_en,
    input  logic signed [ACC_W-1:0] mem_c_rdata,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic                    mem_c_rvalid,
    /* verilator lint_on UNUSEDSIGNAL */
    output logic signed [ACC_W-1:0] mem_c_wdata,
    output logic                    mem_c_wr_en,

    // Performance counter read port. Combinational read of a small
    // counter file; see the PERF_* map below. Counters are cleared when
    // a transaction is accepted, so they always describe the current or
    // most recent run.
    input  logic [4:0]  perf_addr,
    output logic [31:0] perf_rdata
);

  localparam logic [1:0] POLICY_FORCE_WS = 2'b00;
  localparam logic [1:0] POLICY_FORCE_OS = 2'b01;

  localparam int DIM_W   = 16;
  localparam int IDX_W   = $clog2(N_ARR);                 // array row/col index
  localparam int MIDX_W  = $clog2(STREAM_DEPTH);          // index into a_buf/c_buf
  localparam int MCNT_W  = $clog2(STREAM_DEPTH + 1);      // holds count 0..STREAM_DEPTH
  localparam int TCNT_W  = $clog2(N_ARR + 1);             // holds tile size 0..N_ARR
  localparam int CCNT_W  = $clog2(STREAM_DEPTH + 2*N_ARR + 1);
  localparam int MCAND_W = CCNT_W + 1;                    // signed activation index

  // Largest dimension that keeps every flat address inside ADDR_W bits:
  // A is M*K, B is K*N, C is M*N, and 255*256 = 65280 < 65536.
  localparam int MAX_DIM = 255;

  localparam logic [TCNT_W-1:0]         N_ARR_T = TCNT_W'(N_ARR);
  localparam logic [MCNT_W-1:0]         SD_M    = MCNT_W'(STREAM_DEPTH);
  localparam logic signed [MCAND_W-1:0] M_ZERO  = MCAND_W'(0);

  // The two dataflows share S_BFETCH, S_AFETCH, S_WB and S_NEXT: the
  // operand walks and the writeback walk have identical structure in
  // both, differing only in their extents. Only the phases that touch
  // the array differ (WS load/compute vs OS clear/compute/drain).
  typedef enum logic [3:0] {
    S_IDLE, S_BFETCH, S_WLOAD, S_AFETCH, S_COMPUTE, S_WB, S_NEXT, S_DONE,
    S_OS_CLEAR, S_OS_COMPUTE, S_OS_DRAIN
  } state_e;

  // public_flat_rd is a verification hook only (zero hardware cost): it
  // lets the testbench measure the per-phase cycle breakdown directly.
  state_e state /* verilator public_flat_rd */;
  state_e state_n;

  // ---- On-chip buffers ----
  // a_buf is written transposed between the two dataflows: WS indexes it
  // [m][k] because it streams M through a resident weight tile, OS
  // indexes it [k][m] because it streams K through resident
  // accumulators. Same storage, same capacity, different access order.
  logic signed [DATA_W-1:0] a_buf [STREAM_DEPTH][N_ARR];
  logic signed [DATA_W-1:0] b_buf [STREAM_DEPTH][N_ARR]; // b_buf[kk][nn]
  logic signed [ACC_W-1:0]  c_buf [STREAM_DEPTH][N_ARR]; // c_buf[mm][nn]

  // ---- Tile loop origins ----
  logic [DIM_W-1:0] n0, k0, m0;

  // Latched at start: which dataflow this transaction runs.
  logic mode_os;

  // ---- Ragged tile extents, derived combinationally from the origins.
  // Using the true remaining extent (never rounded up to N_ARR) is what
  // makes ragged tiles work without zero-padding: short tiles simply
  // leave array rows/columns inactive.
  //
  // The two dataflows tile the iteration space differently, which is the
  // whole reason their costs differ:
  //   WS maps K x N onto the array and streams M, so K and N are bounded
  //      by N_ARR while M runs up to STREAM_DEPTH per pass.
  //   OS maps M x N onto the array and streams K, so M and N are bounded
  //      by N_ARR while K runs up to STREAM_DEPTH per pass.
  // ----
  logic [DIM_W-1:0]  n_rem, k_rem, m_rem;
  logic [TCNT_W-1:0] n_span;
  logic [MCNT_W-1:0] k_span, m_span;

  assign n_rem  = cfg_n - n0;
  assign k_rem  = cfg_k - k0;
  assign m_rem  = cfg_m - m0;
  assign n_span = (n_rem >= DIM_W'(N_ARR)) ? N_ARR_T : TCNT_W'(n_rem);
  assign k_span = mode_os
                ? ((k_rem >= DIM_W'(STREAM_DEPTH)) ? SD_M : MCNT_W'(k_rem))
                : ((k_rem >= DIM_W'(N_ARR))        ? MCNT_W'(N_ARR) : MCNT_W'(k_rem));
  assign m_span = mode_os
                ? ((m_rem >= DIM_W'(N_ARR))        ? MCNT_W'(N_ARR) : MCNT_W'(m_rem))
                : ((m_rem >= DIM_W'(STREAM_DEPTH)) ? SD_M : MCNT_W'(m_rem));

  // In WS the first K tile writes C directly and every later one
  // read-modify-writes, because partial sums must leave the array
  // between K tiles. In OS the accumulator never leaves the array until
  // the reduction is finished, so C is always a direct write -- that
  // difference is the central hypothesis this project measures.
  logic k_first, c_direct;
  assign k_first  = (k0 == '0);
  assign c_direct = mode_os | k_first;

  // ---- Loop-advance predicates and the next origin values ----
  logic [DIM_W-1:0] m0_nx, k0_nx, n0_nx;
  logic             more_m, more_k, more_n;

  assign m0_nx  = m0 + DIM_W'(m_span);
  assign k0_nx  = k0 + DIM_W'(k_span);
  assign n0_nx  = n0 + DIM_W'(n_span);
  assign more_m = (m0_nx < cfg_m);
  assign more_k = (k0_nx < cfg_k);
  assign more_n = (n0_nx < cfg_n);

  // ---- Tile base addresses (row-major A[M][K], B[K][N], C[M][N]).
  // These multiplies are evaluated once per tile at phase setup, never
  // per element -- the per-element walk below is pure increment. ----
  logic [ADDR_W-1:0] addr_a_base, addr_c_base;
  assign addr_a_base = ADDR_W'(m0 * cfg_k + k0);
  assign addr_c_base = ADDR_W'(m0 * cfg_n + n0);

  // ---- B-tile fetch (k_span rows x n_span columns of B) ----
  // Identical walk in both dataflows; only k_span differs.
  logic [MIDX_W-1:0] bf_kk;
  logic [IDX_W-1:0]  bf_nn;
  logic [ADDR_W-1:0] bf_addr;
  logic              bf_active, bf_last;
  assign bf_last = (bf_kk == MIDX_W'(k_span - MCNT_W'(1))) &&
                   (bf_nn == IDX_W'(n_span - TCNT_W'(1)));

  // ---- Weight load (WS only) ----
  logic [IDX_W-1:0] load_cnt;
  logic             load_last;
  assign load_last = (load_cnt == IDX_W'(k_span[IDX_W:0] - MCNT_W'(1)));

  // ---- A-chunk fetch (m_span rows x k_span columns of A) ----
  logic [MIDX_W-1:0] af_mm;
  logic [MIDX_W-1:0] af_kk;
  logic [ADDR_W-1:0] af_addr;
  logic              af_active, af_last;
  assign af_last = (af_mm == MIDX_W'(m_span - MCNT_W'(1))) &&
                   (af_kk == MIDX_W'(k_span - MCNT_W'(1)));

  // ---- Compute ----
  logic [CCNT_W-1:0] compute_cnt;
  logic [CCNT_W-1:0] compute_last;
  logic [MCNT_W-1:0] col_result_cnt [N_ARR];
  // WS: the last result (m = m_span-1) leaves column n_span-1 at
  //     compute_cnt = (m_span-1) + (n_span-1) + N_ARR. The psum traverses
  //     all N_ARR physical rows even when k_span < N_ARR, because
  //     inactive rows still cost one register stage each -- so the drain
  //     depth is N_ARR, not k_span.
  // OS: A[i][t] and B[t][j] meet at PE(i,j) at compute_cnt = t + i + j,
  //     so the last accumulation is at (k_span-1)+(m_span-1)+(n_span-1).
  //     No pipeline drain is included here; that is the separate
  //     S_OS_DRAIN phase.
  assign compute_last = mode_os
      ? (CCNT_W'(k_span) + CCNT_W'(m_span) + CCNT_W'(n_span) - CCNT_W'(3))
      : (CCNT_W'(m_span) + CCNT_W'(n_span) + CCNT_W'(N_ARR) - CCNT_W'(2));

  // ---- OS accumulator drain ----
  // Always N_ARR cycles: the chain is physically N_ARR wide, so column 0
  // needs N_ARR shifts to reach the east edge regardless of n_span. On
  // drain cycle d the east edge presents the accumulator that started in
  // column N_ARR-1-d.
  logic [IDX_W-1:0] drain_cnt;
  logic             drain_last;
  assign drain_last = (drain_cnt == IDX_W'(N_ARR-1));

  // ---- Writeback / read-modify-write ----
  logic [MIDX_W-1:0]      wb_mm;
  logic [IDX_W-1:0]       wb_nn;
  logic [ADDR_W-1:0]      wb_addr;
  logic                   wb_active, wb_phase, wb_last_elem, wb_final;
  logic signed [ACC_W-1:0] wb_sum;
  // The writeback walk covers the output tile: m_span x n_span in both
  // dataflows.
  assign wb_last_elem = (wb_mm == MIDX_W'(m_span - MCNT_W'(1))) &&
                        (wb_nn == IDX_W'(n_span - TCNT_W'(1)));
  // A direct write completes in its single cycle; an RMW completes only
  // after its write half (phase 1). Non-pipelined by design for v1.
  assign wb_final = c_direct ? (wb_phase == 1'b0) : (wb_phase == 1'b1);

  // ---- cfg validation ----
  logic policy_supported;
  assign policy_supported = (cfg_policy == POLICY_FORCE_WS) ||
                            (cfg_policy == POLICY_FORCE_OS);

  logic cfg_valid;
  assign cfg_valid = (cfg_m != '0) && (cfg_n != '0) && (cfg_k != '0) &&
                     (cfg_m <= DIM_W'(MAX_DIM)) && (cfg_n <= DIM_W'(MAX_DIM)) &&
                     (cfg_k <= DIM_W'(MAX_K)) &&
                     policy_supported;

  // ---- Array interconnect ----
  // Phase encoding shared with the PE: 0 idle, 1 WS load, 2 WS compute,
  // 3 OS clear, 4 OS compute, 5 OS drain.
  logic [2:0] phase;
  assign phase = (state == S_WLOAD)      ? 3'd1 :
                 (state == S_COMPUTE)    ? 3'd2 :
                 (state == S_OS_CLEAR)   ? 3'd3 :
                 (state == S_OS_COMPUTE) ? 3'd4 :
                 (state == S_OS_DRAIN)   ? 3'd5 : 3'd0;

  localparam int MAC_CNT_W = $clog2(N_ARR*N_ARR + 1);

  logic signed [DATA_W-1:0] a_in_arr       [N_ARR];
  logic                     a_valid_in_arr [N_ARR];
  logic signed [DATA_W-1:0] operand_in_arr       [N_ARR];
  logic                     operand_valid_in_arr [N_ARR];
  logic signed [ACC_W-1:0]  psum_out_arr       [N_ARR];
  logic                     psum_valid_out_arr [N_ARR];
  logic signed [ACC_W-1:0]  drain_out_arr      [N_ARR];
  logic [MAC_CNT_W-1:0]     active_macs;

  pe_array #(.N_ARR(N_ARR), .DATA_W(DATA_W), .ACC_W(ACC_W)) u_array (
    .clk           (clk),
    .rst_n         (rst_n),
    .phase         (phase),
    .n_active      (n_span),
    .a_in          (a_in_arr),
    .a_valid_in    (a_valid_in_arr),
    .operand_in       (operand_in_arr),
    .operand_valid_in (operand_valid_in_arr),
    .psum_out      (psum_out_arr),
    .psum_valid_out(psum_valid_out_arr),
    .drain_out     (drain_out_arr),
    .active_macs   (active_macs)
  );

  genvar gr;
  generate
    for (gr = 0; gr < N_ARR; gr++) begin : g_act_feed
      // Both dataflows inject activations at the west edge, but index
      // them differently.
      //
      // WS: array row r holds reduction index k = r, and row r presents
      //     A[m0+m][k0+r] at compute_cnt == m + r. Ragged-K needs no
      //     padding because rows at or past k_span never fire and simply
      //     pass the partial sum through.
      // OS: array row i holds output row i, and row i presents
      //     A[m0+i][k0+t] at compute_cnt == t + i. Ragged-M needs no
      //     padding because rows at or past m_span never fire.
      logic signed [MCAND_W-1:0] idx_ws, idx_os;
      logic                      row_ws, row_os, sel_ws, sel_os;

      assign idx_ws = $signed({1'b0, compute_cnt}) - $signed(MCAND_W'(gr));
      assign idx_os = idx_ws;  // same skew relation, different meaning
      assign row_ws = (MCNT_W'(gr) < k_span);
      assign row_os = (MCNT_W'(gr) < m_span);

      assign sel_ws = (state == S_COMPUTE) && row_ws &&
                      (idx_ws >= M_ZERO) &&
                      (idx_ws < $signed(MCAND_W'(m_span)));
      assign sel_os = (state == S_OS_COMPUTE) && row_os &&
                      (idx_os >= M_ZERO) &&
                      (idx_os < $signed(MCAND_W'(k_span)));

      assign a_valid_in_arr[gr] = sel_ws || sel_os;
      // WS reads a_buf[m][k=row]; OS reads a_buf[t][i=row].
      assign a_in_arr[gr] = sel_ws ? a_buf[idx_ws[MIDX_W-1:0]][gr] :
                            sel_os ? a_buf[idx_os[MIDX_W-1:0]][gr] : '0;
    end

    for (gr = 0; gr < N_ARR; gr++) begin : g_weight_feed
      // WS load: reverse-row-order injection over the tile's true
      // height, so after k_span shifts b_buf[kk] sits in array row kk.
      // Rows at or past k_span keep stale weights, harmless precisely
      // because they are never activated -- this is why ragged K needs
      // no zero-fill of the weight tile.
      //
      // OS compute: the same port streams B[k0+t][n0+j] south into
      // column j at compute_cnt == t + j, one weight per cycle, so the
      // activation from the west and the weight from the north meet at
      // PE(i,j) on the same cycle.
      logic signed [MCAND_W-1:0] t_os;
      logic                      col_os, os_stream;

      assign t_os      = $signed({1'b0, compute_cnt}) - $signed(MCAND_W'(gr));
      assign col_os    = (TCNT_W'(gr) < n_span);
      assign os_stream = (state == S_OS_COMPUTE) && col_os &&
                         (t_os >= M_ZERO) &&
                         (t_os < $signed(MCAND_W'(k_span)));

      assign operand_valid_in_arr[gr] = os_stream;
      assign operand_in_arr[gr] =
          os_stream ? b_buf[t_os[MIDX_W-1:0]][gr]
                    : b_buf[MIDX_W'(k_span - MCNT_W'(1)) - MIDX_W'(load_cnt)][gr];
    end
  endgenerate

  // ---- Result capture ----
  generate
    for (gr = 0; gr < N_ARR; gr++) begin : g_capture
      logic col_in_tile, col_capture;
      assign col_in_tile = (TCNT_W'(gr) < n_span);
      assign col_capture = (state == S_COMPUTE) && psum_valid_out_arr[gr] &&
                           col_in_tile && (col_result_cnt[gr] < m_span);

      // Re-armed on every entry to COMPUTE, not just at reset: this
      // counter is the per-tile result index, so it must restart for
      // each tile. Leaving it to run would park it at m_chunk, making
      // the `< m_chunk` guard false forever and silently freezing c_buf
      // at the first tile's results.
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          col_result_cnt[gr] <= '0;
        end else if (state != S_COMPUTE) begin
          col_result_cnt[gr] <= '0;
        end else if (col_capture) begin
          c_buf[col_result_cnt[gr][MIDX_W-1:0]][gr] <= psum_out_arr[gr];
          col_result_cnt[gr] <= col_result_cnt[gr] + 1'b1;
        end
      end

      // Result-skew invariant, asserted rather than assumed. Holds for
      // ragged tiles too: the psum drain depth is N_ARR regardless of
      // k_tile. int' casts keep this verification-only expression in
      // plain 32-bit signed arithmetic.
      assert property (
        @(posedge clk) disable iff (!rst_n)
          col_capture |->
            (int'(col_result_cnt[gr]) == int'(compute_cnt) - gr - N_ARR)
      ) else $error("column %0d result timing mismatch vs WS formula", gr);
    end
  endgenerate

  // ---- OS drain capture ----
  // On drain cycle d the east edge of row r presents the accumulator
  // that started in column N_ARR-1-d, so results arrive in reverse
  // column order. Columns at or past n_span were cleared and never
  // accumulated; they are captured harmlessly and simply not written
  // back.
  logic [IDX_W-1:0] drain_col;
  assign drain_col = IDX_W'(N_ARR-1) - drain_cnt;

  generate
    for (gr = 0; gr < N_ARR; gr++) begin : g_drain_capture
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          // c_buf is fully rewritten before each writeback; no reset
          // value is needed or useful here.
        end else if (state == S_OS_DRAIN && (MCNT_W'(gr) < m_span)) begin
          c_buf[MIDX_W'(gr)][drain_col] <= drain_out_arr[gr];
        end
      end
    end
  endgenerate

  // ---- Main FSM ----
  always_comb begin
    state_n = state;
    unique case (state)
      S_IDLE: if (start && cfg_valid)
                state_n = (cfg_policy == POLICY_FORCE_OS) ? S_OS_CLEAR : S_BFETCH;

      // Shared operand fetch: the next state depends only on the mode.
      S_BFETCH:  if (bf_active && bf_last)
                   state_n = mode_os ? S_AFETCH : S_WLOAD;
      S_WLOAD:   if (load_last)                       state_n = S_AFETCH;
      S_AFETCH:  if (af_active && af_last)
                   state_n = mode_os ? S_OS_COMPUTE : S_COMPUTE;

      S_COMPUTE: if (compute_cnt == compute_last)     state_n = S_WB;

      // OS keeps the accumulators resident across K chunks: another
      // chunk means going straight back for operands with no clear and
      // no drain, which is exactly why OS never spills partial sums.
      S_OS_CLEAR:   state_n = S_BFETCH;
      S_OS_COMPUTE: if (compute_cnt == compute_last)
                      state_n = more_k ? S_BFETCH : S_OS_DRAIN;
      S_OS_DRAIN:   if (drain_last)                   state_n = S_WB;

      S_WB:      if (wb_active && wb_last_elem && wb_final) state_n = S_NEXT;

      S_NEXT:    state_n = mode_os
                   ? ((more_n || more_m) ? S_OS_CLEAR : S_DONE)
                   : (more_m ? S_AFETCH :
                      (more_k || more_n) ? S_BFETCH : S_DONE);
      S_DONE:    state_n = S_IDLE;
      default:   state_n = S_IDLE;
    endcase
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state       <= S_IDLE;
      n0          <= '0;
      k0          <= '0;
      m0          <= '0;
      bf_kk       <= '0;
      bf_nn       <= '0;
      bf_addr     <= '0;
      bf_active   <= 1'b0;
      load_cnt    <= '0;
      af_mm       <= '0;
      af_kk       <= '0;
      af_addr     <= '0;
      af_active   <= 1'b0;
      compute_cnt <= '0;
      wb_mm       <= '0;
      wb_nn       <= '0;
      wb_addr     <= '0;
      wb_phase    <= 1'b0;
      wb_sum      <= '0;
      wb_active   <= 1'b0;
      drain_cnt   <= '0;
      mode_os     <= 1'b0;
      error       <= 1'b0;
    end else begin
      state <= state_n;
      error <= 1'b0;

      unique case (state)
        S_IDLE: begin
          n0 <= '0;
          k0 <= '0;
          m0 <= '0;
          if (start && cfg_valid) begin
            // Latch the dataflow for the whole transaction. Every span
            // and every state transition keys off this, so it must not
            // move once a run is under way.
            mode_os   <= (cfg_policy == POLICY_FORCE_OS);
            // First tile is (n0,k0) = (0,0), so its B base address is 0.
            bf_kk     <= '0;
            bf_nn     <= '0;
            bf_addr   <= '0;
            bf_active <= 1'b1;
          end
          if (start && !cfg_valid) error <= 1'b1;
        end

        // OS zeroes every accumulator once per output tile, then falls
        // through to the shared operand fetch.
        S_OS_CLEAR: begin
          bf_kk     <= '0;
          bf_nn     <= '0;
          bf_addr   <= ADDR_W'(k0 * cfg_n + n0);
          bf_active <= 1'b1;
        end

        // Walk the B tile in row-major order. As in every fetch phase
        // here, the pre-edge counters are exactly the coordinates of the
        // request that was on the bus during the cycle now ending, so
        // mem_b_rdata at this edge is that element's response and no
        // shadow address register is needed.
        S_BFETCH: begin
          if (bf_active) begin
            b_buf[bf_kk][bf_nn] <= mem_b_rdata;
            if (bf_nn == IDX_W'(n_span - TCNT_W'(1))) begin
              bf_nn   <= '0;
              // Skip to the same column origin on the next B row.
              bf_addr <= bf_addr + ADDR_W'(cfg_n) - ADDR_W'(n_span) + ADDR_W'(1);
              if (bf_last) bf_active <= 1'b0;
              else         bf_kk     <= bf_kk + 1'b1;
            end else begin
              bf_nn   <= bf_nn + 1'b1;
              bf_addr <= bf_addr + ADDR_W'(1);
            end
            // OS skips the weight load and goes straight to A, so set
            // up the A walk as the B walk finishes.
            if (bf_last && mode_os) begin
              af_mm     <= '0;
              af_kk     <= '0;
              af_addr   <= addr_a_base;
              af_active <= 1'b1;
            end
          end
          load_cnt <= '0;
        end

        S_WLOAD: begin
          // Saturate rather than wrap, so a redundant LOAD cycle re-latches
          // the correct final row instead of re-injecting the wrong one.
          if (!load_last) load_cnt <= load_cnt + 1'b1;
          if (load_last) begin
            af_mm     <= '0;
            af_kk     <= '0;
            af_addr   <= addr_a_base;
            af_active <= 1'b1;
          end
        end

        S_AFETCH: begin
          if (af_active) begin
            // WS stores A as [m][k]; OS stores it transposed as [k][m],
            // because OS streams the reduction index through the array
            // while WS streams the output-row index.
            if (mode_os) a_buf[af_kk][af_mm[IDX_W-1:0]] <= mem_a_rdata;
            else         a_buf[af_mm][af_kk[IDX_W-1:0]] <= mem_a_rdata;

            if (af_kk == MIDX_W'(k_span - MCNT_W'(1))) begin
              af_kk   <= '0;
              // Next A row: advance by K, back to this tile's k origin.
              af_addr <= af_addr + ADDR_W'(cfg_k) - ADDR_W'(k_span) + ADDR_W'(1);
              if (af_last) af_active <= 1'b0;
              else         af_mm     <= af_mm + 1'b1;
            end else begin
              af_kk   <= af_kk + 1'b1;
              af_addr <= af_addr + ADDR_W'(1);
            end
          end
          compute_cnt <= '0;
        end

        S_COMPUTE: begin
          compute_cnt <= compute_cnt + 1'b1;
          if (compute_cnt == compute_last) begin
            wb_mm     <= '0;
            wb_nn     <= '0;
            wb_phase  <= 1'b0;
            wb_addr   <= addr_c_base;
            wb_active <= 1'b1;
          end
        end

        S_OS_COMPUTE: begin
          compute_cnt <= compute_cnt + 1'b1;
          if (compute_cnt == compute_last) begin
            drain_cnt <= '0;
            if (more_k) begin
              // Another K chunk for the same output tile. The
              // accumulators stay exactly where they are -- no clear,
              // no drain, no C traffic. This is the property that makes
              // OS cheap for deep reductions.
              k0        <= k0_nx;
              bf_kk     <= '0;
              bf_nn     <= '0;
              bf_addr   <= ADDR_W'(k0_nx * cfg_n + n0);
              bf_active <= 1'b1;
            end
          end
        end

        S_OS_DRAIN: begin
          if (!drain_last) drain_cnt <= drain_cnt + 1'b1;
          if (drain_last) begin
            wb_mm     <= '0;
            wb_nn     <= '0;
            wb_phase  <= 1'b0;
            wb_addr   <= addr_c_base;
            wb_active <= 1'b1;
          end
        end

        S_WB: begin
          if (wb_active) begin
            if (!c_direct && (wb_phase == 1'b0)) begin
              // Read half: mem_c_rdata at this edge answers the read
              // issued during the cycle now ending.
              wb_sum   <= mem_c_rdata + c_buf[wb_mm][wb_nn];
              wb_phase <= 1'b1;
            end else begin
              wb_phase <= 1'b0;
              if (wb_nn == IDX_W'(n_span - TCNT_W'(1))) begin
                wb_nn   <= '0;
                wb_addr <= wb_addr + ADDR_W'(cfg_n) - ADDR_W'(n_span) + ADDR_W'(1);
                if (wb_last_elem) wb_active <= 1'b0;
                else              wb_mm     <= wb_mm + 1'b1;
              end else begin
                wb_nn   <= wb_nn + 1'b1;
                wb_addr <= wb_addr + ADDR_W'(1);
              end
            end
          end
        end

        S_NEXT: if (mode_os) begin
          // OS advances the output tile only; K always restarts, since
          // the whole reduction completed inside the array.
          k0 <= '0;
          if (more_n) begin
            n0 <= n0_nx;
          end else begin
            n0 <= '0;
            m0 <= m0_nx;
          end
        end else begin
          if (more_m) begin
            // Weights stay resident: straight back to the A fetch with
            // no B refetch and no weight reload.
            m0        <= m0_nx;
            af_mm     <= '0;
            af_kk     <= '0;
            af_addr   <= ADDR_W'(m0_nx * cfg_k + k0);
            af_active <= 1'b1;
          end else begin
            m0 <= '0;
            bf_kk     <= '0;
            bf_nn     <= '0;
            bf_active <= 1'b1;
            if (more_k) begin
              k0      <= k0_nx;
              bf_addr <= ADDR_W'(k0_nx * cfg_n + n0);
            end else begin
              k0      <= '0;
              n0      <= n0_nx;
              // k0 restarts at 0, so the B base collapses to the new n0.
              bf_addr <= ADDR_W'(n0_nx);
            end
          end
        end

        default: ;
      endcase
    end
  end

  assign busy = (state != S_IDLE) && (state != S_DONE);
  assign done = (state == S_DONE);

  assign mem_a_addr  = af_addr;
  assign mem_a_rd_en = (state == S_AFETCH) && af_active;

  assign mem_b_addr  = bf_addr;
  assign mem_b_rd_en = (state == S_BFETCH) && bf_active;

  assign mem_c_addr  = wb_addr;
  assign mem_c_rd_en = (state == S_WB) && wb_active && !c_direct && (wb_phase == 1'b0);
  assign mem_c_wdata = c_direct ? c_buf[wb_mm][wb_nn] : wb_sum;
  assign mem_c_wr_en = (state == S_WB) && wb_active &&
                       (c_direct ? (wb_phase == 1'b0) : (wb_phase == 1'b1));

  // ---- Performance counters ----
  //
  // The accelerator reports its own behavior rather than relying on
  // external waveform analysis. Everything here is measured from actual
  // hardware activity (state occupancy, bus enables, the array's own
  // valid popcount), never predicted from the schedule -- otherwise the
  // counters would just restate the model they exist to check.
  //
  // Cleared on each accepted start, so a read after `done` describes
  // exactly the run that just finished.
  localparam logic [4:0]
      PERF_TOTAL_CYCLES = 5'd0,  PERF_COMPUTE_CYCLES = 5'd1,
      PERF_BFETCH       = 5'd2,  PERF_WLOAD          = 5'd3,
      PERF_AFETCH       = 5'd4,  PERF_WB             = 5'd5,
      PERF_STALL        = 5'd6,  PERF_WEIGHT_TILES   = 5'd7,
      PERF_M_CHUNKS     = 5'd8,  PERF_MAC_OPS        = 5'd9,
      PERF_MAC_SLOTS    = 5'd10, PERF_BYTES_A        = 5'd11,
      PERF_BYTES_B      = 5'd12, PERF_C_WRITES       = 5'd13,
      PERF_C_RMW        = 5'd14, PERF_ERRORS         = 5'd15,
      PERF_OS_CLEAR     = 5'd16, PERF_OS_DRAIN       = 5'd17;

  logic [31:0] cnt_total, cnt_compute, cnt_bfetch, cnt_wload, cnt_afetch;
  logic [31:0] cnt_wb, cnt_weight_tiles, cnt_m_chunks, cnt_mac;
  logic [31:0] cnt_bytes_a, cnt_bytes_b, cnt_c_writes, cnt_c_rmw, cnt_errors;
  logic [31:0] cnt_os_clear, cnt_os_drain;

  logic perf_clear;
  assign perf_clear = (state == S_IDLE) && start && cfg_valid;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      cnt_total        <= '0; cnt_compute  <= '0; cnt_bfetch   <= '0;
      cnt_wload        <= '0; cnt_afetch   <= '0; cnt_wb       <= '0;
      cnt_weight_tiles <= '0; cnt_m_chunks <= '0; cnt_mac      <= '0;
      cnt_bytes_a      <= '0; cnt_bytes_b  <= '0; cnt_c_writes <= '0;
      cnt_c_rmw        <= '0; cnt_errors   <= '0;
      cnt_os_clear     <= '0; cnt_os_drain <= '0;
    end else if (perf_clear) begin
      // A new transaction resets everything except the error count,
      // which is a lifetime tally and deliberately survives.
      cnt_total        <= '0; cnt_compute  <= '0; cnt_bfetch   <= '0;
      cnt_wload        <= '0; cnt_afetch   <= '0; cnt_wb       <= '0;
      cnt_weight_tiles <= '0; cnt_m_chunks <= '0; cnt_mac      <= '0;
      cnt_bytes_a      <= '0; cnt_bytes_b  <= '0; cnt_c_writes <= '0;
      cnt_c_rmw        <= '0;
      cnt_os_clear     <= '0; cnt_os_drain <= '0;
    end else begin
      if (busy) cnt_total <= cnt_total + 32'd1;

      unique case (state)
        S_BFETCH: begin
          cnt_bfetch <= cnt_bfetch + 32'd1;
          // One B element lands per active fetch cycle.
          if (bf_active) cnt_bytes_b <= cnt_bytes_b + 32'd1;
        end
        S_WLOAD: begin
          cnt_wload <= cnt_wload + 32'd1;
          if (load_last) cnt_weight_tiles <= cnt_weight_tiles + 32'd1;
        end
        S_AFETCH: begin
          cnt_afetch <= cnt_afetch + 32'd1;
          if (af_active) cnt_bytes_a <= cnt_bytes_a + 32'd1;
        end
        // Both dataflows report compute occupancy and MACs through the
        // same counters, so the two are directly comparable. Clear and
        // drain are OS-only overhead and are counted separately as
        // stall, not as compute.
        S_COMPUTE, S_OS_COMPUTE: begin
          cnt_compute <= cnt_compute + 32'd1;
          cnt_mac     <= cnt_mac + 32'(active_macs);
        end
        S_OS_CLEAR:  cnt_os_clear <= cnt_os_clear + 32'd1;
        S_OS_DRAIN:  cnt_os_drain <= cnt_os_drain + 32'd1;
        S_WB:   cnt_wb <= cnt_wb + 32'd1;
        S_NEXT: cnt_m_chunks <= cnt_m_chunks + 32'd1;
        S_IDLE: if (start && !cfg_valid) cnt_errors <= cnt_errors + 32'd1;
        default: ;
      endcase

      if (mem_c_wr_en) cnt_c_writes <= cnt_c_writes + 32'd1;
      if (mem_c_rd_en) cnt_c_rmw    <= cnt_c_rmw + 32'd1;
    end
  end

  // MAC slots = compute cycles x array size. mac_ops/mac_slots is the
  // array's true occupancy while computing; mac_ops/(total x size) is
  // its occupancy over the whole transaction.
  always_comb begin
    unique case (perf_addr)
      PERF_TOTAL_CYCLES:   perf_rdata = cnt_total;
      PERF_COMPUTE_CYCLES: perf_rdata = cnt_compute;
      PERF_BFETCH:         perf_rdata = cnt_bfetch;
      PERF_WLOAD:          perf_rdata = cnt_wload;
      PERF_AFETCH:         perf_rdata = cnt_afetch;
      PERF_WB:             perf_rdata = cnt_wb;
      // Everything the array is not computing: fetch, load, writeback,
      // and (OS only) clear and drain.
      PERF_STALL:          perf_rdata = cnt_total - cnt_compute;
      PERF_OS_CLEAR:       perf_rdata = cnt_os_clear;
      PERF_OS_DRAIN:       perf_rdata = cnt_os_drain;
      PERF_WEIGHT_TILES:   perf_rdata = cnt_weight_tiles;
      PERF_M_CHUNKS:       perf_rdata = cnt_m_chunks;
      PERF_MAC_OPS:        perf_rdata = cnt_mac;
      PERF_MAC_SLOTS:      perf_rdata = cnt_compute * 32'(N_ARR*N_ARR);
      PERF_BYTES_A:        perf_rdata = cnt_bytes_a;
      PERF_BYTES_B:        perf_rdata = cnt_bytes_b;
      PERF_C_WRITES:       perf_rdata = cnt_c_writes;
      PERF_C_RMW:          perf_rdata = cnt_c_rmw;
      PERF_ERRORS:         perf_rdata = cnt_errors;
      default:             perf_rdata = 32'd0;
    endcase
  end

  // ---- Architectural invariants ----

  // The array may never report more concurrent MACs than it has PEs.
  property p_macs_bounded;
    @(posedge clk) disable iff (!rst_n)
      active_macs <= MAC_CNT_W'(N_ARR*N_ARR);
  endproperty
  assert property (p_macs_bounded)
    else $error("active_macs exceeds the number of PEs");

  // ...and must report none at all outside a compute phase, which is
  // what makes the MAC total trustworthy as a measure of real work.
  // Both dataflows have a compute state and both feed the same counter.
  property p_no_macs_outside_compute;
    @(posedge clk) disable iff (!rst_n)
      ((state != S_COMPUTE) && (state != S_OS_COMPUTE))
        |-> (active_macs == MAC_CNT_W'(0));
  endproperty
  assert property (p_no_macs_outside_compute)
    else $error("active_macs asserted outside COMPUTE");

  // A C read and a C write must never be requested in the same cycle:
  // the v1 RMW path is deliberately non-pipelined.
  property p_c_rd_wr_exclusive;
    @(posedge clk) disable iff (!rst_n) !(mem_c_rd_en && mem_c_wr_en);
  endproperty
  assert property (p_c_rd_wr_exclusive)
    else $error("C read and write asserted in the same cycle");

  // The first K tile must never read C back -- there is no pre-zeroing
  // pass, so a read there would consume undefined memory.
  property p_no_rmw_on_first_k;
    @(posedge clk) disable iff (!rst_n) (c_direct |-> !mem_c_rd_en);
  endproperty
  assert property (p_no_rmw_on_first_k)
    else $error("read-modify-write attempted on a direct-write tile");

  // Output-stationary must never read C back at all: the entire
  // reduction happens inside the array, so a C read would mean a
  // partial sum escaped. This is the invariant the whole WS-vs-OS
  // comparison rests on.
  property p_os_never_reads_c;
    @(posedge clk) disable iff (!rst_n) (mode_os |-> !mem_c_rd_en);
  endproperty
  assert property (p_os_never_reads_c)
    else $error("OS issued a C read; accumulator escaped the array");

  // The dataflow must not change while a transaction is in flight.
  // Both cycles must be busy: mode_os is latched on the very edge that
  // raises busy, so requiring stability on the first busy cycle would
  // flag that legitimate latch.
  property p_mode_stable_while_busy;
    @(posedge clk) disable iff (!rst_n)
      (busy && $past(busy)) |-> $stable(mode_os);
  endproperty
  assert property (p_mode_stable_while_busy)
    else $error("dataflow mode changed mid-transaction");

  // Weights must stay resident while M chunks stream: no weight load may
  // occur between a compute and the next compute of the same tile.
  property p_no_reload_within_m_loop;
    @(posedge clk) disable iff (!rst_n)
      ((state == S_NEXT) && more_m) |=> (state != S_BFETCH) && (state != S_WLOAD);
  endproperty
  assert property (p_no_reload_within_m_loop)
    else $error("weights reloaded while streaming M chunks of the same tile");

  // Spans must always be non-zero, within STREAM_DEPTH, and -- for
  // whichever dimensions that dataflow maps spatially -- within the
  // physical array.
  property p_tile_extents_legal;
    @(posedge clk) disable iff (!rst_n)
      busy |-> (n_span >= TCNT_W'(1)) && (n_span <= N_ARR_T) &&
               (k_span >= MCNT_W'(1)) && (k_span <= SD_M) &&
               (m_span >= MCNT_W'(1)) && (m_span <= SD_M);
  endproperty
  assert property (p_tile_extents_legal)
    else $error("illegal tile extent");

  // The dimension each dataflow maps onto the array must fit in it:
  // WS maps K x N, OS maps M x N.
  property p_spatial_span_fits;
    @(posedge clk) disable iff (!rst_n)
      busy |-> (mode_os ? (m_span <= MCNT_W'(N_ARR)) : (k_span <= MCNT_W'(N_ARR)));
  endproperty
  assert property (p_spatial_span_fits)
    else $error("spatially-mapped span exceeds the array");

  property p_busy_not_with_done;
    @(posedge clk) disable iff (!rst_n) !(busy && done);
  endproperty
  assert property (p_busy_not_with_done)
    else $error("busy and done asserted simultaneously");

endmodule
