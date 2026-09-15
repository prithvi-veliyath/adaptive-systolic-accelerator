// N_ARR x N_ARR weight-stationary mesh.
module pe_array #(
    parameter int N_ARR  = 4,
    parameter int DATA_W = 8,
    parameter int ACC_W  = 32,
    localparam int MAC_CNT_W = $clog2(N_ARR*N_ARR + 1),
    localparam int TCNT_W    = $clog2(N_ARR + 1)
) (
    input  logic clk,
    input  logic rst_n,
    input  logic [1:0] phase,

    // Width of the current output tile. Columns at or beyond this index
    // hold weights from a previous tile and must not be allowed to
    // multiply: see the gate below.
    input  logic [TCNT_W-1:0] n_active,

    input  logic signed [DATA_W-1:0] a_in       [N_ARR],
    input  logic                     a_valid_in [N_ARR],
    input  logic signed [DATA_W-1:0] operand_in [N_ARR],

    output logic signed [ACC_W-1:0] psum_out       [N_ARR],
    output logic                    psum_valid_out [N_ARR],

    // Number of PEs performing a real multiply-accumulate this cycle.
    // Measured from the array's own valid inputs rather than predicted
    // from the schedule, so the utilization figure it feeds reflects
    // what the hardware actually did.
    output logic [MAC_CNT_W-1:0] active_macs
);

  logic signed [DATA_W-1:0] a_wire         [N_ARR][N_ARR+1];
  logic                     a_valid_wire   [N_ARR][N_ARR+1];
  logic                     a_valid_gated  [N_ARR][N_ARR];
  logic signed [DATA_W-1:0] op_wire        [N_ARR+1][N_ARR];
  logic signed [ACC_W-1:0]  psum_wire      [N_ARR+1][N_ARR];
  logic                     psum_valid_wire[N_ARR+1][N_ARR];

  genvar r, c;
  generate
    for (r = 0; r < N_ARR; r++) begin : g_west_edge
      assign a_wire[r][0]       = a_in[r];
      assign a_valid_wire[r][0] = a_valid_in[r];
    end

    for (c = 0; c < N_ARR; c++) begin : g_north_edge
      assign op_wire[0][c]         = operand_in[c];
      assign psum_wire[0][c]       = '0;
      assign psum_valid_wire[0][c] = 1'b0;
    end

    for (r = 0; r < N_ARR; r++) begin : g_row
      for (c = 0; c < N_ARR; c++) begin : g_col
        // Ragged-N column gate. Without it the activation valid keeps
        // travelling east past the end of the tile, and columns holding
        // a previous tile's weights perform multiply-accumulates whose
        // results are simply never captured. That is invisible to a
        // correctness test -- the outputs are still right -- but it
        // burns real MACs and real power on every ragged-N tile.
        // Gating the first inactive column is sufficient, because an
        // inactive PE registers a zero valid and so starves the rest of
        // the row downstream of it.
        assign a_valid_gated[r][c] = a_valid_wire[r][c] && (TCNT_W'(c) < n_active);

        pe #(.DATA_W(DATA_W), .ACC_W(ACC_W)) u_pe (
          .clk           (clk),
          .rst_n         (rst_n),
          .phase         (phase),
          .a_in          (a_wire[r][c]),
          .a_valid_in    (a_valid_gated[r][c]),
          .a_out         (a_wire[r][c+1]),
          .a_valid_out   (a_valid_wire[r][c+1]),
          .operand_in    (op_wire[r][c]),
          .operand_out   (op_wire[r+1][c]),
          .psum_in       (psum_wire[r][c]),
          .psum_valid_in (psum_valid_wire[r][c]),
          .psum_out      (psum_wire[r+1][c]),
          .psum_valid_out(psum_valid_wire[r+1][c])
        );

        // Skew-correctness self-check (rows > 0 only -- row 0 has no
        // real north neighbor, its psum_valid is a fixed boundary 0).
        //
        // Stated as an implication rather than an equality: an *active*
        // PE must be receiving a valid partial sum from the north, which
        // is what catches a broken skew. The converse does not hold once
        // ragged K tiles exist -- rows below k_tile are inactive yet
        // legitimately forward a valid psum downward -- so requiring
        // equality here would fire on correct ragged-tile behavior.
        if (r > 0) begin : g_valid_check
          assert property (
            @(posedge clk) disable iff (!rst_n)
              (phase == 2'd2) |->
                (a_valid_gated[r][c] |-> psum_valid_wire[r][c])
          ) else $error("row-skew mismatch at PE(%0d,%0d)", r, c);
        end
      end
    end

    for (c = 0; c < N_ARR; c++) begin : g_south_edge
      assign psum_out[c]       = psum_wire[N_ARR][c];
      assign psum_valid_out[c] = psum_valid_wire[N_ARR][c];
    end
  endgenerate

  // Popcount of the array's activation-valid inputs, qualified by the
  // COMPUTE phase.
  //
  // Both terms are required. A PE accumulates only when its own valid
  // is asserted AND the array is in COMPUTE -- outside COMPUTE psum_reg
  // does not update at all, so no MAC occurs even though the activation
  // valid registers still hold their last values for one cycle until
  // the inter-tile flush clears them. Counting those would overstate
  // real work during every tile changeover.
  always_comb begin
    active_macs = '0;
    if (phase == 2'd2)
      for (int rr = 0; rr < N_ARR; rr++)
        for (int cc = 0; cc < N_ARR; cc++)
          if (a_valid_gated[rr][cc]) active_macs = active_macs + MAC_CNT_W'(1);
  end

endmodule
