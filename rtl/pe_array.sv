// N_ARR x N_ARR weight-stationary mesh (Milestone 1).
module pe_array #(
    parameter int N_ARR  = 4,
    parameter int DATA_W = 8,
    parameter int ACC_W  = 32
) (
    input  logic clk,
    input  logic rst_n,
    input  logic [1:0] phase,

    input  logic signed [DATA_W-1:0] a_in       [N_ARR],
    input  logic                     a_valid_in [N_ARR],
    input  logic signed [DATA_W-1:0] operand_in [N_ARR],

    output logic signed [ACC_W-1:0] psum_out       [N_ARR],
    output logic                    psum_valid_out [N_ARR]
);

  logic signed [DATA_W-1:0] a_wire         [N_ARR][N_ARR+1];
  logic                     a_valid_wire   [N_ARR][N_ARR+1];
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
        pe #(.DATA_W(DATA_W), .ACC_W(ACC_W)) u_pe (
          .clk           (clk),
          .rst_n         (rst_n),
          .phase         (phase),
          .a_in          (a_wire[r][c]),
          .a_valid_in    (a_valid_wire[r][c]),
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
                (a_valid_wire[r][c] |-> psum_valid_wire[r][c])
          ) else $error("row-skew mismatch at PE(%0d,%0d)", r, c);
        end
      end
    end

    for (c = 0; c < N_ARR; c++) begin : g_south_edge
      assign psum_out[c]       = psum_wire[N_ARR][c];
      assign psum_valid_out[c] = psum_valid_wire[N_ARR][c];
    end
  endgenerate

endmodule
