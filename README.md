# Adaptive Systolic Accelerator

A configurable `N x N` signed-INT8 systolic matrix-multiply accelerator that runs
**two different dataflows on one shared PE array** and picks between them at
runtime based on matrix shape.

> Can a single shared systolic array support both weight-stationary (WS) and
> output-stationary (OS) execution, and can hardware choose the better one per
> workload — with the choice justified by **measured cycle counts** rather than
> intuition?

Yes, and the answer is quantified below. The project is built as an experiment:
the RTL is the instrument, and the deliverable is a measured result.

**Status: all 7 milestones complete.** WS and OS on the shared array, hardware
performance counters, double-buffered operand prefetch, a 56-shape measurement
sweep, and an ADAPTIVE policy derived from that sweep that picks the faster
dataflow on **56 of 56 shapes with zero regret**.

---

## Quick start

Requires Verilator (tested with 5.032) and GTKWave.

```bash
cd sim && make
```

Lints (`-Wall`, warnings are errors), builds, runs the full suite. Ends with
`ALL TESTS PASSED`.

```bash
cd sim && make bench
```

Runs the WS-vs-OS sweep and writes `docs/benchmark.csv`.

---

## The result

Both dataflows run on identical data over a 56-shape grid
(`M ∈ {1,4,16,64}`, `N ∈ {4,16}`, `K ∈ {1,2,4,8,16,32,64}`), each verified
against a golden model.

**WS wins 16, OS wins 28, 12 ties.** The crossover is sharp and systematic, and
the crossover **K rises with M**. That follows directly from the structural
asymmetry between the two dataflows:

- **WS** must spill partial sums out to `C` and read them back once `K` exceeds
  the array height → `2·M·N·(Ktiles−1)` extra memory accesses.
- **OS** keeps the accumulator resident and never touches `C` mid-reduction, but
  re-reads the entire `B` tile once per `M` tile, and pays a clear plus an
  `N_ARR`-cycle drain per output tile.
- **A-operand traffic is identical in both** and cancels exactly.

On `37×10×13` that trade shows up in one line: writeback collapses
**2590 → 370 cycles** while B-fetch rises **130 → 1300**.

### Double buffering

The A fetch is not an FSM phase but an independent prefetch engine, so an
operand fetch overlaps compute and writeback, and the A and B buses run
concurrently. `a_buf` is two-banked: the engine fills one bank while the array
computes from the other.

On `37×10×13` WS this cut **4911 → 4126 cycles (16%)**, with visible A-fetch
dropping `1443 → 658`. OS gains less — its inner loop is K, so with
`K ≤ STREAM_DEPTH` there is nothing to prefetch — which is exactly what shifted
the crossover below.

### Choosing between them

Candidate policies scored against the *measured* winner (oracle = always pick
the faster one):

| Policy | Correct | Total regret | Worst case |
|---|---|---|---|
| always WS | 28/56 | 49630 cycles | 191.8% |
| always OS | 40/56 | 10055 cycles | 133.2% |
| `K > N_ARR` | 52/56 | 3470 cycles | 41.1% |
| cost-difference model (pre-prefetch) | 54/56 | 1140 cycles | 9.0% |
| **cost model + prefetch (shipped)** | **56/56** | **0** | **0.0%** |
| **ADAPTIVE (in hardware)** | **56/56** | **0** | **0.0%** |

The obvious rule — "switch to OS once the reduction outgrows the array" — is not
free. It misses the large-M corner where OS's per-M-tile B refetch and drain
overhead still outweigh the writeback it saves.

The shipped policy compares the cost differences directly:

```
favours OS:  2·M·N·(Ktiles−1)                 WS spills partial sums through C
           + Ntiles·K                         WS reloads weights per (n,k) tile
           + Ntiles·min(M,SD)·K               WS's A fetch is mostly hidden

favours WS:  K·N·(Mtiles−1)                   OS re-reads B once per M tile
           + (N_ARR+1)·Mtiles·Ntiles          OS clear + drain per output tile
           + Ntiles·M·min(K,SD)               OS's A fetch is mostly visible
```

Evaluated once per transaction at start, so the multipliers are a one-shot cost
and not on any critical path.

### The part worth reading twice

The first version of this policy scored **56/56** — and then double buffering
made it **worse**, dropping it to 54/56.

Adding the prefetch engine changed which costs matter. The original model let
A-operand traffic cancel between the dataflows, because both moved the same
elements at the same cost. Once A can overlap other work, that stops being true:
WS hides its A fetch behind compute *and* writeback, while OS can only hide it
behind a much shorter window — and when `K ≤ STREAM_DEPTH` it has no second
chunk to prefetch at all.

Every simpler rule degraded too, and more severely: `K > N_ARR` went from a 5.8%
worst case to **41.1%**.

A heuristic fitted to one microarchitecture does not survive a change to that
microarchitecture. That's the whole argument for deriving the policy from
measurement and re-running the sweep whenever the cost balance moves — and this
repo has the before-and-after to prove it.

---

## Repository layout

```
rtl/
  pe.sv          dual-dataflow PE: stationary weight OR stationary accumulator
  pe_array.sv    N x N mesh, drain chain, MAC activity measurement
  accel_top.sv   tiling control, both dataflows, adaptive policy, counters
sim/
  tb_accel.cpp   golden model, memory model, analytical models, sweep
  Makefile       `make` to test, `make bench` to sweep
docs/
  architecture.md   detailed architecture, derivations, design decisions
  benchmark.csv     sweep results
```

---

## Architecture

### One array, two dataflows

Both dataflows share the multiplier, the activation register, and the
north–south operand register. They differ only in **which register is
stationary** and **where the result leaves**:

| | WS | OS |
|---|---|---|
| Stationary | `weight_reg` | `accumulator_reg` |
| Streams | M through the array | K through the array |
| Spatial map | K × N | M × N |
| Result exits | south, as partial sums | east, via a dedicated drain chain |
| C traffic | 1 write + RMW per extra K tile | exactly 1 write |

`weight_reg` and `accumulator_reg` are deliberately **separate** architectural
state — one shared register would make the two dataflows impossible to reason
about independently.

The drain path is dedicated rather than folded onto the activation path: the
activation path is only `DATA_W` wide and is carrying operands, while a drain
must move a full `ACC_W` accumulator.

The control is genuinely shared too — `S_BFETCH`, `S_AFETCH`, `S_WB` and
`S_NEXT` are common to both, because the operand walks and the writeback walk
have identical structure and differ only in their extents.

### Two properties that are load-bearing and non-obvious

- **Validity passes through inactive PEs** alongside the data. Regenerating it
  locally kills the flag at the first inactive row, and no tile with
  `k_tile < N_ARR` would ever emit a result.
- **The valid pipeline is flushed whenever the array isn't computing.** Without
  it, a valid still in flight at the end of one tile re-enters the next as a
  spurious operand.

### Ragged tiles, no padding

Short tiles use their true extents. Three independent gates (`r < k_span`,
`0 ≤ m < m_span`, `c < n_span`) make them fall out naturally. Rows past the tile
keep stale weights, which is harmless precisely because they never fire.

This is a **measured** property, not a claim about the source: the hardware MAC
counter reports exactly `M·N·K` on every shape tested. A padding implementation
would necessarily report more.

---

## Verification

Every output is checked against a software golden GEMM, and **every phase's
cycle count against independently-written analytical models** of both schedules.

- **Values**: identity, all-negative, zeros, mixed signs, INT8 `127`/`-128`
- **Shapes**: 14 directed WS + 12 directed OS, covering N/K/M tiling, ragged in
  every dimension, `1×1×1`, thin row/column, chunking past `STREAM_DEPTH`,
  deep K, and `37×10×13`
- **120 randomized shapes × both dataflows**, fixed seed
- **Adaptive**: hardware choice checked against measured winners
- **Control**: reset mid-transaction, back-to-back runs, all four policy
  encodings, six illegal shapes
- **Per run**: exact MAC count vs `M·N·K`, exact C write count, no surviving
  poison, every memory access bounds-checked, all counters cross-checked against
  observed occupancy

Twelve SystemVerilog assertions guard architectural invariants. Six have fired
during development; five caught real RTL bugs.

---

## Bugs found and root-caused

Full write-ups in [docs/architecture.md](docs/architecture.md#8-design-decisions-and-bugs-found).
Two are worth highlighting because **no correctness test could have caught
them** — they were found by cross-checking hardware counters:

**Ragged-N tiles burned ~17% of all MACs on discarded work.** The activation
valid kept travelling east past the tile edge, so columns holding the previous
tile's weights kept multiplying. Outputs were always correct. Caught by checking
the MAC counter against `M·N·K` (5763 vs 4810 on `37×10×13`).

**ADAPTIVE silently skipped the accumulator clear.** The next-state decode tested
`cfg_policy == FORCE_OS` literally, so an ADAPTIVE run that *chose* OS entered
through the WS path and never zeroed the accumulators. Results stayed correct by
luck — a completed drain shifts zeros in behind itself. Caught by cross-checking
ADAPTIVE's cycle count against the forced run of the dataflow it reported.

The other seven: FILL address shadow lagging the bus by one cycle; `wb_cnt`
overshoot wrapping on the `done` edge; a weight-stability assertion sampling the
wrong cycle; an unsigned-zero comparison that made a bounds check constant-true;
valid propagation killing ragged-K results; `col_result_cnt` not re-armed per
tile; and the valid pipeline not self-draining between tiles.

Clean under `-Wall` with no width suppressions; the one remaining waiver
(`SYNCASYNCNET`, the standard async-reset + `disable iff` pattern) is scoped to
single port declarations.

---

## Roadmap

| Milestone | Scope |
|---|---|
| **1 — done** | PE, 4×4 array, WS dataflow, exact-fit, FSM, golden-model TB, assertions |
| **2 — done** | Tiling, ragged tiles without padding, multi-K accumulation, C RMW |
| **3 — done** | Hardware performance counters; found the ragged-N MAC waste |
| **4 — done** | Output-stationary dataflow on the shared array + drain path |
| **5 — done** | 56-shape WS vs OS sweep; located the crossover |
| **6 — done** | ADAPTIVE policy derived from the sweep — 56/56, zero regret |
| **7 — done** | Double-buffered prefetch (WS +16%); policy re-derived after it moved the crossover |

### Future work

- Scale to 8×8 and re-run the sweep; the crossover should move again
- Prefetch across tile boundaries, which would help OS (its inner loop often
  runs once, so it rarely prefetches today)
- Pipeline the C read-modify-write path
- Per-tile adaptive selection rather than per-transaction

---

## Known limitations

- C read-modify-write is non-pipelined (2 cycles/element) by deliberate v1 choice
- Prefetch covers the innermost loop only, so a tile's first chunk is never
  hidden
- The adaptive decision is per-transaction, not per-tile
- No accumulator saturation (32-bit wraparound, matching the golden model)
- `M, N ≤ 255` and `K ≤ 256`, so every flat address fits in 16 bits
- Memory model assumes fixed 1-cycle latency, in-order responses, no contention
- The policy is validated on the 56-shape grid above; it is a measured fit to
  this memory model, not a universal law
