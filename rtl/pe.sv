// Milestone 1: weight-stationary-only PE.
// accumulator_reg, drain_reg and a separate operand_reg are intentionally
// absent -- they belong to OS mode, not yet implemented (see spec review).
module pe #(
    parameter int DATA_W = 8,
    parameter int ACC_W  = 32
) (
    input  logic clk,
    input  logic rst_n,

    input  logic [1:0] phase,   // 0=IDLE, 1=LOAD, 2=COMPUTE

    input  logic signed [DATA_W-1:0] a_in,
    input  logic                     a_valid_in,
    output logic signed [DATA_W-1:0] a_out,
    output logic                     a_valid_out,

    input  logic signed [DATA_W-1:0] operand_in,
    output logic signed [DATA_W-1:0] operand_out,

    input  logic signed [ACC_W-1:0] psum_in,
    input  logic                    psum_valid_in,
    output logic signed [ACC_W-1:0] psum_out,
    output logic                    psum_valid_out
);

  localparam logic [1:0] PHASE_LOAD    = 2'd1;
  localparam logic [1:0] PHASE_COMPUTE = 2'd2;

  logic signed [DATA_W-1:0] weight_reg;
  logic signed [DATA_W-1:0] a_reg;
  logic                     a_valid_reg;
  logic signed [ACC_W-1:0]  psum_reg;
  logic                     psum_valid_reg;

  // Weight load: unconditional shift while phase==LOAD. Reverse-row-order
  // injection (a controller/array responsibility) is what makes this
  // settle to the correct value array-wide after exactly N_ARR cycles.
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      weight_reg <= '0;
    else if (phase == PHASE_LOAD)
      weight_reg <= operand_in;
  end
  assign operand_out = weight_reg;

  // Activation shift, west -> east, active only during COMPUTE.
  //
  // Validity is cleared whenever the array is not computing, so the
  // pipeline is empty at the start of every tile. Without this flush a
  // valid left in flight when one tile's COMPUTE ends would re-enter the
  // next tile's COMPUTE as a spurious operand. (Data is not cleared:
  // without its valid it can never be consumed, and leaving it saves the
  // flops.) This is what makes "tiles flush before the next one begins"
  // an enforced property rather than an assumption.
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      a_reg       <= '0;
      a_valid_reg <= 1'b0;
    end else if (phase == PHASE_COMPUTE) begin
      a_reg       <= a_in;
      a_valid_reg <= a_valid_in;
    end else begin
      a_valid_reg <= 1'b0;
    end
  end
  assign a_out       = a_reg;
  assign a_valid_out = a_valid_reg;

  // Dedicated WS partial-sum path. Gated by a_valid_in -- the temporal
  // analog of spatial row_active (no ragged tiles exist in Milestone 1,
  // so this only ever gates pipeline fill/drain cycles).
  logic signed [2*DATA_W-1:0] product;
  logic signed [ACC_W-1:0]    product_ext;
  logic signed [ACC_W-1:0]    mac_result;

  assign product     = a_in * weight_reg;
  assign product_ext = ACC_W'(product);
  assign mac_result  = psum_in + product_ext;

  // Data and validity must pass through an inactive PE identically.
  //
  // An inactive row (a ragged K tile leaves rows k_tile..N_ARR-1 with no
  // real weights) forwards psum_in untouched, so it must forward the
  // accompanying psum_valid_in too. Regenerating validity from the local
  // a_valid_in instead would kill the valid flag at the first inactive
  // row, and since the array is physically N_ARR tall the bottom row
  // would never emit a valid result for any tile with k_tile < N_ARR.
  //
  // For a full tile the two formulations are equivalent: the activation
  // and partial-sum wavefronts coincide, so a_valid_in is true at row r
  // exactly when psum_valid_in is.
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      psum_reg       <= '0;
      psum_valid_reg <= 1'b0;
    end else if (phase == PHASE_COMPUTE) begin
      psum_reg       <= a_valid_in ? mac_result : psum_in;
      psum_valid_reg <= a_valid_in | psum_valid_in;
    end else begin
      // Flushed between tiles, same reasoning as the activation path.
      // Pass-through validity means the psum pipeline does not
      // self-drain by the last COMPUTE cycle, so an explicit flush is
      // required rather than merely convenient.
      psum_valid_reg <= 1'b0;
    end
  end
  assign psum_out       = psum_reg;
  assign psum_valid_out = psum_valid_reg;

  // Invariant: weight_reg must never change outside LOAD.
  //
  // Note the antecedent uses $past(phase), not the current-cycle phase.
  // weight_reg's flip-flop is gated by phase sampled at the SAME edge as
  // its own update (the standard synchronous read-before-write ordering),
  // so the final LOAD-phase write becomes visible in weight_reg's sampled
  // value on the first cycle where `phase` has already advanced to
  // COMPUTE. Gating on current-cycle phase would therefore flag that
  // legitimate last write as a violation, one cycle after the fact.
  // $past(phase) checks the phase that actually gated the write.
  property p_weight_stable_outside_load;
    @(posedge clk) disable iff (!rst_n)
      ($past(phase) != PHASE_LOAD) |-> $stable(weight_reg);
  endproperty
  assert property (p_weight_stable_outside_load)
    else $error("weight_reg changed outside LOAD phase");

endmodule
