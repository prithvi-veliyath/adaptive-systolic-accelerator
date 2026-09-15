// Dual-dataflow processing element: weight-stationary and
// output-stationary on the same physical hardware.
//
// The two dataflows share every datapath element -- the multiplier, the
// activation register, the north-south operand register -- and differ
// only in which register is stationary and where the result leaves:
//
//   WS: weight_reg is stationary; partial sums flow north -> south and
//       exit the south edge.
//   OS: accumulator_reg is stationary; weights stream north -> south
//       through the same operand register, and results leave later via
//       a dedicated east-flowing drain path.
//
// weight_reg and accumulator_reg are deliberately separate architectural
// state. Folding them into one register would make the two dataflows
// impossible to reason about independently, and would prevent a future
// design from switching mode without a full flush.
module pe #(
    parameter int DATA_W = 8,
    parameter int ACC_W  = 32
) (
    input  logic clk,
    // Async FF reset that is also referenced in assertion `disable iff`
    // clauses -- the standard safe pattern, and the only SYNCASYNCNET
    // waiver here. Scoped to this port alone.
    /* verilator lint_off SYNCASYNCNET */
    input  logic rst_n,
    /* verilator lint_on SYNCASYNCNET */

    input  logic [2:0] phase,

    input  logic signed [DATA_W-1:0] a_in,
    input  logic                     a_valid_in,
    output logic signed [DATA_W-1:0] a_out,
    output logic                     a_valid_out,

    // North-south operand path. Carries stationary weights during a WS
    // load, and streaming weights during an OS compute.
    input  logic signed [DATA_W-1:0] operand_in,
    input  logic                     operand_valid_in,
    output logic signed [DATA_W-1:0] operand_out,
    output logic                     operand_valid_out,

    // WS partial-sum path, north -> south.
    input  logic signed [ACC_W-1:0] psum_in,
    input  logic                    psum_valid_in,
    output logic signed [ACC_W-1:0] psum_out,
    output logic                    psum_valid_out,

    // OS accumulator drain path, west -> east. Dedicated rather than
    // folded onto the activation path: the activation path is only
    // DATA_W wide and carries operands, while a drain must move a full
    // ACC_W accumulator, and overlaying the two would make the drain
    // ordering depend on activation timing.
    input  logic signed [ACC_W-1:0] drain_in,
    output logic signed [ACC_W-1:0] drain_out
);

  localparam logic [2:0] PHASE_LOAD       = 3'd1;  // WS weight load
  localparam logic [2:0] PHASE_COMPUTE    = 3'd2;  // WS compute
  localparam logic [2:0] PHASE_OS_CLEAR   = 3'd3;  // zero the accumulators
  localparam logic [2:0] PHASE_OS_COMPUTE = 3'd4;  // OS accumulate
  localparam logic [2:0] PHASE_OS_DRAIN   = 3'd5;  // shift accumulators east

  logic signed [DATA_W-1:0] weight_reg;
  logic                     weight_valid_reg;
  logic signed [DATA_W-1:0] a_reg;
  logic                     a_valid_reg;
  logic signed [ACC_W-1:0]  psum_reg;
  logic                     psum_valid_reg;
  logic signed [ACC_W-1:0]  accumulator_reg;

  // North-south operand register, shared by both dataflows.
  //
  // WS: unconditional shift while phase==LOAD. Reverse-row-order
  //     injection (an array/controller responsibility) is what makes
  //     this settle array-wide after exactly k_tile cycles, after which
  //     it is stationary for the whole compute.
  // OS: the same register becomes a streaming pipeline stage, moving one
  //     weight south per cycle for the duration of the reduction.
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      weight_reg       <= '0;
      weight_valid_reg <= 1'b0;
    end else if (phase == PHASE_LOAD) begin
      weight_reg       <= operand_in;
      weight_valid_reg <= 1'b0;          // WS does not use operand validity
    end else if (phase == PHASE_OS_COMPUTE) begin
      weight_reg       <= operand_in;
      weight_valid_reg <= operand_valid_in;
    end else if (phase != PHASE_COMPUTE) begin
      // Flush the streaming validity between tiles, but hold weight_reg
      // itself: WS relies on it staying stationary across its COMPUTE.
      weight_valid_reg <= 1'b0;
    end
  end
  assign operand_out       = weight_reg;
  assign operand_valid_out = weight_valid_reg;

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
    end else if (phase == PHASE_COMPUTE || phase == PHASE_OS_COMPUTE) begin
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

  // OS multiplies the two operands arriving *this* cycle -- the
  // activation from the west and the weight from the north -- rather
  // than against a stationary register. Both wavefronts are skewed so
  // that A[i][t] and B[t][j] meet at PE(i,j) on the same cycle.
  logic signed [2*DATA_W-1:0] os_product;
  logic signed [ACC_W-1:0]    os_product_ext;
  logic                       os_mac_en;

  assign os_product     = a_in * operand_in;
  assign os_product_ext = ACC_W'(os_product);
  assign os_mac_en      = a_valid_in && operand_valid_in;

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

  // ---- Output-stationary accumulator and drain ----
  //
  // The accumulator is the stationary state in OS: it holds one output
  // element for the entire reduction over K, however many K chunks that
  // takes. That is the whole point of the dataflow -- the partial sum
  // never leaves the PE, so it is never spilled to memory and never
  // read back, which is exactly the cost WS pays through C
  // read-modify-write.
  //
  // CLEAR zeroes it once per output tile; COMPUTE accumulates; DRAIN
  // turns the row into an east-moving shift register so the results can
  // be walked out of the east edge one column per cycle.
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      accumulator_reg <= '0;
    else if (phase == PHASE_OS_CLEAR)
      accumulator_reg <= '0;
    else if (phase == PHASE_OS_COMPUTE && os_mac_en)
      accumulator_reg <= accumulator_reg + os_product_ext;
    else if (phase == PHASE_OS_DRAIN)
      accumulator_reg <= drain_in;
  end
  assign drain_out = accumulator_reg;

  // Invariant: weight_reg must never change outside LOAD.
  //
  // Note the antecedent uses $past(phase), not the current-cycle phase.
  // weight_reg's flip-flop is gated by phase sampled at the SAME edge as
  // its own update (the standard synchronous read-before-write ordering),
  // so the final LOAD-phase write becomes visible in weight_reg's sampled
  // value on the first cycle where `phase` has already advanced to
  // COMPUTE. Gating on current-cycle phase would therefore flag that
  // legitimate last write as a violation, one cycle after the fact.
  // OS is exempt because there weight_reg is a streaming pipeline stage
  // rather than stationary state.
  property p_weight_stable_outside_load;
    @(posedge clk) disable iff (!rst_n)
      (($past(phase) != PHASE_LOAD) && ($past(phase) != PHASE_OS_COMPUTE))
        |-> $stable(weight_reg);
  endproperty
  assert property (p_weight_stable_outside_load)
    else $error("weight_reg changed outside LOAD phase");

  // Invariant: the OS accumulator is stationary during its reduction --
  // it may only change on a real MAC, and never at all during a drain
  // except by taking its west neighbour's value.
  property p_accum_stable_without_mac;
    @(posedge clk) disable iff (!rst_n)
      (($past(phase) == PHASE_OS_COMPUTE) && !$past(os_mac_en))
        |-> $stable(accumulator_reg);
  endproperty
  assert property (p_accum_stable_without_mac)
    else $error("accumulator_reg changed without a valid MAC");

endmodule
