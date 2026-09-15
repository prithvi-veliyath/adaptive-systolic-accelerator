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
    input  logic [3:0]  perf_addr,
    output logic [31:0] perf_rdata
);

  localparam logic [1:0] POLICY_FORCE_WS = 2'b00;

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

  typedef enum logic [2:0] {
    S_IDLE, S_BFETCH, S_WLOAD, S_AFETCH, S_COMPUTE, S_WB, S_NEXT, S_DONE
  } state_e;

  // public_flat_rd is a verification hook only (zero hardware cost): it
  // lets the testbench measure the per-phase cycle breakdown directly.
  state_e state /* verilator public_flat_rd */;
  state_e state_n;

  // ---- On-chip buffers ----
  logic signed [DATA_W-1:0] a_buf [STREAM_DEPTH][N_ARR]; // a_buf[mm][kk]
  logic signed [DATA_W-1:0] b_buf [N_ARR][N_ARR];        // b_buf[kk][nn]
  logic signed [ACC_W-1:0]  c_buf [STREAM_DEPTH][N_ARR]; // c_buf[mm][nn]

  // ---- Tile loop origins ----
  logic [DIM_W-1:0] n0, k0, m0;

  // ---- Ragged tile extents, derived combinationally from the origins.
  // Using the true remaining extent (never rounded up to N_ARR) is what
  // makes ragged tiles work without zero-padding: short tiles simply
  // leave array rows/columns inactive. ----
  logic [DIM_W-1:0]  n_rem, k_rem, m_rem;
  logic [TCNT_W-1:0] n_tile, k_tile;
  logic [MCNT_W-1:0] m_chunk;

  assign n_rem   = cfg_n - n0;
  assign k_rem   = cfg_k - k0;
  assign m_rem   = cfg_m - m0;
  assign n_tile  = (n_rem >= DIM_W'(N_ARR))        ? N_ARR_T : TCNT_W'(n_rem);
  assign k_tile  = (k_rem >= DIM_W'(N_ARR))        ? N_ARR_T : TCNT_W'(k_rem);
  assign m_chunk = (m_rem >= DIM_W'(STREAM_DEPTH)) ? SD_M    : MCNT_W'(m_rem);

  // First K tile writes C directly; every later one read-modify-writes.
  // No pre-zeroing pass over C is performed.
  logic k_first;
  assign k_first = (k0 == '0);

  // ---- Loop-advance predicates and the next origin values ----
  logic [DIM_W-1:0] m0_nx, k0_nx, n0_nx;
  logic             more_m, more_k, more_n;

  assign m0_nx  = m0 + DIM_W'(m_chunk);
  assign k0_nx  = k0 + DIM_W'(k_tile);
  assign n0_nx  = n0 + DIM_W'(n_tile);
  assign more_m = (m0_nx < cfg_m);
  assign more_k = (k0_nx < cfg_k);
  assign more_n = (n0_nx < cfg_n);

  // ---- Tile base addresses (row-major A[M][K], B[K][N], C[M][N]).
  // These multiplies are evaluated once per tile at phase setup, never
  // per element -- the per-element walk below is pure increment. ----
  logic [ADDR_W-1:0] addr_a_base, addr_c_base;
  assign addr_a_base = ADDR_W'(m0 * cfg_k + k0);
  assign addr_c_base = ADDR_W'(m0 * cfg_n + n0);

  // ---- B-tile fetch ----
  logic [IDX_W-1:0]  bf_kk, bf_nn;
  logic [ADDR_W-1:0] bf_addr;
  logic              bf_active, bf_last;
  assign bf_last = (bf_kk == IDX_W'(k_tile - TCNT_W'(1))) &&
                   (bf_nn == IDX_W'(n_tile - TCNT_W'(1)));

  // ---- Weight load ----
  logic [IDX_W-1:0] load_cnt;
  logic             load_last;
  assign load_last = (load_cnt == IDX_W'(k_tile - TCNT_W'(1)));

  // ---- A-chunk fetch ----
  logic [MIDX_W-1:0] af_mm;
  logic [IDX_W-1:0]  af_kk;
  logic [ADDR_W-1:0] af_addr;
  logic              af_active, af_last;
  assign af_last = (af_mm == MIDX_W'(m_chunk - MCNT_W'(1))) &&
                   (af_kk == IDX_W'(k_tile - TCNT_W'(1)));

  // ---- Compute ----
  logic [CCNT_W-1:0] compute_cnt;
  logic [CCNT_W-1:0] compute_last;
  logic [MCNT_W-1:0] col_result_cnt [N_ARR];
  // Last result (m = m_chunk-1) leaves column n_tile-1 at
  // compute_cnt = (m_chunk-1) + (n_tile-1) + N_ARR. The psum traverses
  // all N_ARR physical rows even when k_tile < N_ARR, because inactive
  // rows still cost one register stage each -- so the drain depth is
  // N_ARR, not k_tile.
  assign compute_last = CCNT_W'(m_chunk) + CCNT_W'(n_tile) + CCNT_W'(N_ARR) - CCNT_W'(2);

  // ---- Writeback / read-modify-write ----
  logic [MIDX_W-1:0]      wb_mm;
  logic [IDX_W-1:0]       wb_nn;
  logic [ADDR_W-1:0]      wb_addr;
  logic                   wb_active, wb_phase, wb_last_elem, wb_final;
  logic signed [ACC_W-1:0] wb_sum;
  assign wb_last_elem = (wb_mm == MIDX_W'(m_chunk - MCNT_W'(1))) &&
                        (wb_nn == IDX_W'(n_tile - TCNT_W'(1)));
  // A direct write completes in its single cycle; an RMW completes only
  // after its write half (phase 1). Non-pipelined by design for v1.
  assign wb_final = k_first ? (wb_phase == 1'b0) : (wb_phase == 1'b1);

  // ---- cfg validation ----
  logic cfg_valid;
  assign cfg_valid = (cfg_m != '0) && (cfg_n != '0) && (cfg_k != '0) &&
                     (cfg_m <= DIM_W'(MAX_DIM)) && (cfg_n <= DIM_W'(MAX_DIM)) &&
                     (cfg_k <= DIM_W'(MAX_K)) &&
                     (cfg_policy == POLICY_FORCE_WS);

  // ---- Array interconnect ----
  logic [1:0] phase;
  assign phase = (state == S_WLOAD)   ? 2'd1 :
                 (state == S_COMPUTE) ? 2'd2 : 2'd0;

  localparam int MAC_CNT_W = $clog2(N_ARR*N_ARR + 1);

  logic signed [DATA_W-1:0] a_in_arr       [N_ARR];
  logic                     a_valid_in_arr [N_ARR];
  logic signed [DATA_W-1:0] operand_in_arr [N_ARR];
  logic signed [ACC_W-1:0]  psum_out_arr       [N_ARR];
  logic                     psum_valid_out_arr [N_ARR];
  logic [MAC_CNT_W-1:0]     active_macs;

  pe_array #(.N_ARR(N_ARR), .DATA_W(DATA_W), .ACC_W(ACC_W)) u_array (
    .clk           (clk),
    .rst_n         (rst_n),
    .phase         (phase),
    .a_in          (a_in_arr),
    .a_valid_in    (a_valid_in_arr),
    .operand_in    (operand_in_arr),
    .psum_out      (psum_out_arr),
    .psum_valid_out(psum_valid_out_arr),
    .active_macs   (active_macs)
  );

  genvar gr;
  generate
    for (gr = 0; gr < N_ARR; gr++) begin : g_act_feed
      // Array row r holds reduction index kk = r. Row r presents
      // A[m0+m][k0+r] at compute_cnt == m + r.
      //
      // Two independent gates produce ragged-K behavior with no padding:
      //   - r < k_tile    : rows beyond the true tile height never fire,
      //                     so they pass psum through untouched
      //   - 0 <= m < m_chunk : temporal gate for pipeline fill/drain and
      //                     for a short final M chunk
      logic signed [MCAND_W-1:0] m_candidate;
      logic                      row_in_tile;
      assign m_candidate = $signed({1'b0, compute_cnt}) - $signed(MCAND_W'(gr));
      assign row_in_tile = (TCNT_W'(gr) < k_tile);
      assign a_valid_in_arr[gr] = (state == S_COMPUTE) && row_in_tile &&
                                  (m_candidate >= M_ZERO) &&
                                  (m_candidate < $signed(MCAND_W'(m_chunk)));
      assign a_in_arr[gr] = a_valid_in_arr[gr] ?
                            a_buf[m_candidate[MIDX_W-1:0]][gr] : '0;
    end

    for (gr = 0; gr < N_ARR; gr++) begin : g_weight_feed
      // Reverse-row-order injection over the tile's true height: after
      // k_tile shifts, b_buf[kk] sits in array row kk. Rows at or below
      // k_tile retain stale weights, which is harmless precisely because
      // they are never activated -- this is why ragged K needs no
      // zero-fill of the weight tile.
      assign operand_in_arr[gr] = b_buf[IDX_W'(k_tile - TCNT_W'(1)) - load_cnt][gr];
    end
  endgenerate

  // ---- Result capture ----
  generate
    for (gr = 0; gr < N_ARR; gr++) begin : g_capture
      logic col_in_tile, col_capture;
      assign col_in_tile = (TCNT_W'(gr) < n_tile);
      assign col_capture = (state == S_COMPUTE) && psum_valid_out_arr[gr] &&
                           col_in_tile && (col_result_cnt[gr] < m_chunk);

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

  // ---- Main FSM ----
  always_comb begin
    state_n = state;
    unique case (state)
      S_IDLE:    if (start && cfg_valid)              state_n = S_BFETCH;
      S_BFETCH:  if (bf_active && bf_last)            state_n = S_WLOAD;
      S_WLOAD:   if (load_last)                       state_n = S_AFETCH;
      S_AFETCH:  if (af_active && af_last)            state_n = S_COMPUTE;
      S_COMPUTE: if (compute_cnt == compute_last)     state_n = S_WB;
      S_WB:      if (wb_active && wb_last_elem && wb_final) state_n = S_NEXT;
      S_NEXT:    state_n = more_m ? S_AFETCH :
                           (more_k || more_n) ? S_BFETCH : S_DONE;
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
            // First tile is (n0,k0) = (0,0), so its B base address is 0.
            bf_kk     <= '0;
            bf_nn     <= '0;
            bf_addr   <= '0;
            bf_active <= 1'b1;
          end
          if (start && !cfg_valid) error <= 1'b1;
        end

        // Walk the B tile in row-major order. As in every fetch phase
        // here, the pre-edge counters are exactly the coordinates of the
        // request that was on the bus during the cycle now ending, so
        // mem_b_rdata at this edge is that element's response and no
        // shadow address register is needed.
        S_BFETCH: begin
          if (bf_active) begin
            b_buf[bf_kk][bf_nn] <= mem_b_rdata;
            if (bf_nn == IDX_W'(n_tile - TCNT_W'(1))) begin
              bf_nn   <= '0;
              // Skip to the same column origin on the next B row.
              bf_addr <= bf_addr + ADDR_W'(cfg_n) - ADDR_W'(n_tile) + ADDR_W'(1);
              if (bf_last) bf_active <= 1'b0;
              else         bf_kk     <= bf_kk + 1'b1;
            end else begin
              bf_nn   <= bf_nn + 1'b1;
              bf_addr <= bf_addr + ADDR_W'(1);
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
            a_buf[af_mm][af_kk] <= mem_a_rdata;
            if (af_kk == IDX_W'(k_tile - TCNT_W'(1))) begin
              af_kk   <= '0;
              // Next A row: advance by K, back to this tile's k origin.
              af_addr <= af_addr + ADDR_W'(cfg_k) - ADDR_W'(k_tile) + ADDR_W'(1);
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

        S_WB: begin
          if (wb_active) begin
            if (!k_first && (wb_phase == 1'b0)) begin
              // Read half: mem_c_rdata at this edge answers the read
              // issued during the cycle now ending.
              wb_sum   <= mem_c_rdata + c_buf[wb_mm][wb_nn];
              wb_phase <= 1'b1;
            end else begin
              wb_phase <= 1'b0;
              if (wb_nn == IDX_W'(n_tile - TCNT_W'(1))) begin
                wb_nn   <= '0;
                wb_addr <= wb_addr + ADDR_W'(cfg_n) - ADDR_W'(n_tile) + ADDR_W'(1);
                if (wb_last_elem) wb_active <= 1'b0;
                else              wb_mm     <= wb_mm + 1'b1;
              end else begin
                wb_nn   <= wb_nn + 1'b1;
                wb_addr <= wb_addr + ADDR_W'(1);
              end
            end
          end
        end

        S_NEXT: begin
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
  assign mem_c_rd_en = (state == S_WB) && wb_active && !k_first && (wb_phase == 1'b0);
  assign mem_c_wdata = k_first ? c_buf[wb_mm][wb_nn] : wb_sum;
  assign mem_c_wr_en = (state == S_WB) && wb_active &&
                       (k_first ? (wb_phase == 1'b0) : (wb_phase == 1'b1));

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
  localparam logic [3:0]
      PERF_TOTAL_CYCLES = 4'd0,  PERF_COMPUTE_CYCLES = 4'd1,
      PERF_BFETCH       = 4'd2,  PERF_WLOAD          = 4'd3,
      PERF_AFETCH       = 4'd4,  PERF_WB             = 4'd5,
      PERF_STALL        = 4'd6,  PERF_WEIGHT_TILES   = 4'd7,
      PERF_M_CHUNKS     = 4'd8,  PERF_MAC_OPS        = 4'd9,
      PERF_MAC_SLOTS    = 4'd10, PERF_BYTES_A        = 4'd11,
      PERF_BYTES_B      = 4'd12, PERF_C_WRITES       = 4'd13,
      PERF_C_RMW        = 4'd14, PERF_ERRORS         = 4'd15;

  logic [31:0] cnt_total, cnt_compute, cnt_bfetch, cnt_wload, cnt_afetch;
  logic [31:0] cnt_wb, cnt_weight_tiles, cnt_m_chunks, cnt_mac;
  logic [31:0] cnt_bytes_a, cnt_bytes_b, cnt_c_writes, cnt_c_rmw, cnt_errors;

  logic perf_clear;
  assign perf_clear = (state == S_IDLE) && start && cfg_valid;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      cnt_total        <= '0; cnt_compute  <= '0; cnt_bfetch   <= '0;
      cnt_wload        <= '0; cnt_afetch   <= '0; cnt_wb       <= '0;
      cnt_weight_tiles <= '0; cnt_m_chunks <= '0; cnt_mac      <= '0;
      cnt_bytes_a      <= '0; cnt_bytes_b  <= '0; cnt_c_writes <= '0;
      cnt_c_rmw        <= '0; cnt_errors   <= '0;
    end else if (perf_clear) begin
      // A new transaction resets everything except the error count,
      // which is a lifetime tally and deliberately survives.
      cnt_total        <= '0; cnt_compute  <= '0; cnt_bfetch   <= '0;
      cnt_wload        <= '0; cnt_afetch   <= '0; cnt_wb       <= '0;
      cnt_weight_tiles <= '0; cnt_m_chunks <= '0; cnt_mac      <= '0;
      cnt_bytes_a      <= '0; cnt_bytes_b  <= '0; cnt_c_writes <= '0;
      cnt_c_rmw        <= '0;
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
        S_COMPUTE: begin
          cnt_compute <= cnt_compute + 32'd1;
          cnt_mac     <= cnt_mac + 32'(active_macs);
        end
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
      PERF_STALL:          perf_rdata = cnt_total - cnt_compute;
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

  // ...and must report none at all outside COMPUTE, which is what makes
  // the MAC total trustworthy as a measure of real work.
  property p_no_macs_outside_compute;
    @(posedge clk) disable iff (!rst_n)
      (state != S_COMPUTE) |-> (active_macs == MAC_CNT_W'(0));
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
    @(posedge clk) disable iff (!rst_n) (k_first |-> !mem_c_rd_en);
  endproperty
  assert property (p_no_rmw_on_first_k)
    else $error("read-modify-write attempted on the first K tile");

  // Weights must stay resident while M chunks stream: no weight load may
  // occur between a compute and the next compute of the same tile.
  property p_no_reload_within_m_loop;
    @(posedge clk) disable iff (!rst_n)
      ((state == S_NEXT) && more_m) |=> (state != S_BFETCH) && (state != S_WLOAD);
  endproperty
  assert property (p_no_reload_within_m_loop)
    else $error("weights reloaded while streaming M chunks of the same tile");

  // Tile extents must always be within the physical array.
  property p_tile_extents_legal;
    @(posedge clk) disable iff (!rst_n)
      busy |-> (k_tile >= TCNT_W'(1)) && (k_tile <= N_ARR_T) &&
               (n_tile >= TCNT_W'(1)) && (n_tile <= N_ARR_T) &&
               (m_chunk >= MCNT_W'(1)) && (m_chunk <= SD_M);
  endproperty
  assert property (p_tile_extents_legal)
    else $error("illegal tile extent");

  property p_busy_not_with_done;
    @(posedge clk) disable iff (!rst_n) !(busy && done);
  endproperty
  assert property (p_busy_not_with_done)
    else $error("busy and done asserted simultaneously");

endmodule
