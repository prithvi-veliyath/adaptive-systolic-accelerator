# Adaptive Systolic Accelerator

A configurable `N x N` signed-INT8 systolic matrix-multiply accelerator built to
investigate one question:

> Can a **single shared** PE array support both weight-stationary (WS) and
> output-stationary (OS) execution, with a shape-aware scheduler picking the
> better dataflow per tile — and can that choice be justified by **measured**
> hardware cycle counts rather than intuition?

The project is framed as an experiment, not just an implementation. The
deliverable is a measured answer about dataflow selection, with the RTL as the
instrument.

**Status: Milestone 2 complete** — tiled weight-stationary GEMM for arbitrary
`M x N x K`, ragged tiles with no zero-padding, multi-K accumulation via C
read-modify-write. 134 shapes verified against a golden model, zero lint
warnings.

---

## Quick start

Requires Verilator (tested with 5.032) and GTKWave.

```bash
cd sim && make
```

Lints (`-Wall`, warnings are errors), builds, and runs the full suite. Expected
output ends with `ALL TESTS PASSED`.

```bash
gtkwave sim/waveform.vcd
```

---

## The first real result

Milestone 2 already produced a finding that motivates the rest of the project.
Measured on the spec's own `37 x 10 x 13` workload:

| Phase | Cycles | Share |
|---|---|---|
| B fetch | 130 | 2.6% |
| Weight load | 39 | 0.8% |
| **A fetch** | **1443** | **29.4%** |
| Compute | 672 | 13.7% |
| **C writeback (RMW)** | **2590** | **52.7%** |
| Loop overhead | 36 | 0.7% |
| **Total** | **4911** | |

**The array is idle 86% of the time**, and the single largest consumer is not
compute or operand fetch — it is **writing partial sums back out to C**.

Weight-stationary execution has to spill and re-read partial sums through memory
whenever `K` exceeds the array height: with 4 K tiles, three of them pay a
2-cycle read-modify-write per output element. **Output-stationary execution
keeps the accumulator resident in the PE and pays none of that cost.**

So the project's central hypothesis is now a concrete, testable prediction
rather than a hunch:

> WS should lose to OS as K grows relative to `N_ARR`, with the crossover near
> the point where accumulated C read-modify-write traffic exceeds OS's fixed
> drain cost.

Milestones 4–5 measure exactly that. Milestone 6 turns the measurement into the
adaptive policy.

---

## Repository layout

```
rtl/
  pe.sv          processing element: weight_reg + activation + dedicated psum path
  pe_array.sv    N x N mesh with edge wiring and skew self-checks
  accel_top.sv   tiling control FSM, on-chip buffers, memory interfaces
sim/
  tb_accel.cpp   golden model, memory model, analytical cycle model, tests
  Makefile       verilate + build + run
docs/
  architecture.md   detailed architecture, timing derivations, design decisions
```

---

## Architecture

### Processing element

Separate architectural state rather than one ambiguous shared register:
`weight_reg` (stationary), `a_reg` (west -> east), `psum_reg` (north -> south, on
a dedicated psum path). The MAC is `psum_in + (a_in * weight_reg)` sign-extended
to 32 bits.

Weights shift vertically during LOAD; partial sums flow vertically during
COMPUTE. Those two uses are **temporally disjoint**, which is what lets one
shared physical array host a second dataflow later instead of duplicating the
mesh.

Two properties in the PE are load-bearing and non-obvious:

- **Validity passes through inactive PEs** alongside the data. Regenerating it
  locally would kill the flag at the first inactive row, and no tile with
  `k_tile < N_ARR` would ever produce a result.
- **The valid pipeline is flushed whenever the array is not computing**, so each
  tile starts empty. Without it, a valid still in flight at the end of one tile
  re-enters the next tile as a spurious operand.

### Tiling

```
for n_tile:
  for k_tile:
    fetch B tile -> weight load
    for m_chunk:
      fetch A chunk -> compute -> writeback
```

The **m loop is innermost so weights stay resident** across every M chunk —
reloading per chunk would defeat weight-stationary execution entirely. An
assertion enforces that no weight reload occurs inside the M loop.

**Ragged tiles use their true extents; nothing is ever zero-padded.** Three
independent gates (`r < k_tile`, `0 <= m < m_chunk`, `c < n_tile`) make short
tiles fall out naturally. Rows past `k_tile` keep stale weights, which is
harmless precisely because they never fire.

**No pre-zeroing pass over C.** The first K tile writes directly; later K tiles
read-modify-write. The testbench poisons C beforehand so a missed write shows up
as surviving poison rather than a plausible-looking zero.

### Key timing relations (asserted in RTL, not assumed)

- row `r` presents `A[m0+m][k0+r]` at `compute_cnt == m + r`
- column `c` emits its `m`-th result at `compute_cnt == m + c + N_ARR`
- compute length: `m_chunk + n_tile + N_ARR - 1`

The drain depth is `N_ARR` even for short tiles, because an inactive row still
costs a register stage. That is why the result formula is unaffected by
raggedness — and why the frozen `2*K_tile` cost model underestimates ragged K
tiles, a discrepancy carried forward into the Milestone 5 comparison rather than
quietly corrected.

---

## Verification

Every output compared against a software golden GEMM, and **every phase's cycle
count compared against an independently-written analytical model** of the WS
schedule. All 134 shapes match the model exactly.

- **Values**: identity, all-negative, zeros, mixed signs, INT8 `127`/`-128`
- **Shapes**: N tiling, K tiling, both, ragged N, ragged K, ragged in all three
  dimensions, `1x1x1`, thin row/column, M chunking at and past `STREAM_DEPTH`,
  deep K, and the spec's `37x10x13`
- **120 randomized shapes**, fixed seed for reproducibility
- **Control**: reset mid-transaction, back-to-back runs without reset, all four
  policy encodings, six illegal shapes
- **Per run**: exact C write count, no surviving poison, all memory accesses
  bounds-checked

Eight SystemVerilog assertions guard architectural invariants. Four have fired
during development; three caught real RTL bugs.

`ADAPTIVE` deliberately errors rather than falling back to WS or OS — faking
adaptivity before the scheduler exists would corrupt every later measurement.

---

## Bugs found and root-caused

Full write-ups in [docs/architecture.md](docs/architecture.md#8-design-decisions-and-bugs-found).

1. **FILL address shadow lagged one cycle** — every memory response written to
   the previous address's slot. Predicted the exact corrupted output matrix from
   the hypothesis and confirmed it bit-for-bit before changing code.
2. **`wb_cnt` overshoot** — wrapped to 0 on the cycle `done` asserted.
3. **Weight-stability assertion sampled the wrong cycle** — fixed via
   `$past(phase)`; invariant corrected, not weakened.
4. **Unsigned-zero comparison** — constant-true bounds check that no functional
   test caught. Found only after removing a file-scope lint suppression.
5. **Valid propagation killed ragged-K results** — bottom row never asserted
   valid for short tiles.
6. **`col_result_cnt` not re-armed per tile** — `c_buf` froze at the first
   tile's results; diagnosed from the signature that test 1 passed and all
   others returned stale data.
7. **Valid pipeline did not self-drain between tiles** — caught by the
   result-skew assertion rather than by a data mismatch.

The design is clean under `-Wall` with no width suppressions; the single
remaining waiver (`SYNCASYNCNET`, the standard async-reset + `disable iff`
pattern) is scoped to one port declaration.

---

## Roadmap

| Milestone | Scope |
|---|---|
| **1 — done** | PE, 4x4 array, WS dataflow, exact-fit 4x4x4, FSM, memory model, golden-model TB, assertions |
| **2 — done** | Tiling, ragged tiles without padding, multi-K accumulation, C read-modify-write, analytical cycle model |
| 3 | Hardware performance counters — the accelerator reports its own utilization |
| 4 | Output-stationary dataflow on the shared array, dedicated east-flowing drain path |
| 5 | Measure WS vs OS across the shape space; find the real crossover |
| 6 | Derive the adaptive heuristic **from the measurements**, then implement `ADAPTIVE` |
| 7 | Double buffering, to attack the 29% operand-fetch stall measured above |
| later | Scale to 8x8 |

---

## Known limitations

- WS only; OS and ADAPTIVE are unimplemented and error out by design
- No double buffering — operand fetch is fully serial with compute
- C read-modify-write is non-pipelined (2 cycles/element) by deliberate v1 choice
- No accumulator saturation (32-bit wraparound, matching the golden model)
- `M, N <= 255` and `K <= 256`, so every flat address fits in 16 bits
- Memory model assumes fixed 1-cycle latency, in-order responses, no contention;
  the shadow-register-free fetch path depends on that contract and is commented
  as such
