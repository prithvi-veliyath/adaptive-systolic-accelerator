# Architecture

Reference for the adaptive systolic accelerator. Tracks what is **actually
implemented and verified**; sections marked *(planned)* are deliberately not
built yet.

**Current state: all 7 milestones — both dataflows on a shared array, hardware
counters, double-buffered operand prefetch, a measured WS-vs-OS sweep, and an
adaptive policy derived from it and then re-derived after the prefetch engine
changed the cost balance.**

---

## 1. Design thesis

A single configurable `N x N` systolic PE array supporting **both**
weight-stationary (WS) and output-stationary (OS) execution, with a shape-aware
scheduler choosing the better dataflow per tile.

The interesting question is not "can this multiply matrices" but:

> For a given tile shape `(M_tile, N_tile, K_tile)` on a fixed `N x N` array,
> which dataflow wins, by how much, and can hardware pick correctly at runtime?

Answering that honestly requires measured cycle counts from both dataflows on a
*shared* physical array — which is why the array is shared rather than
duplicated, and why the adaptive policy will be derived from measurements
rather than asserted up front.

Milestone 2 has already produced the first real evidence bearing on that
question. See §9.

---

## 2. Frozen parameters

| Parameter | Value | Notes |
|---|---|---|
| `N_ARR` | 4 | parameterized; 8x8 is a later scaling exercise |
| `DATA_W` | 8 | signed INT8 operands |
| `ACC_W` | 32 | signed accumulation, **no saturation** |
| `MAX_K` | 256 | validated |
| `STREAM_DEPTH` | 16 | max M rows resident per chunk |

`N_ARR` is assumed a power of two. `MAX_DIM = 255` bounds M and N so every flat
address fits in `ADDR_W = 16` bits (`255 * 256 = 65280 < 65536`).

---

## 3. Dataflow policy encoding

| Encoding | Policy | Status |
|---|---|---|
| `2'b00` | `FORCE_WS` | implemented |
| `2'b01` | `FORCE_OS` | implemented |
| `2'b10` | `ADAPTIVE` | implemented (§9) |
| `2'b11` | `RESERVED` | rejected: `error`, `busy` stays low |

Through Milestones 1–5, `ADAPTIVE` deliberately raised `error` rather than
silently degrading to WS or OS. Faking adaptivity before the scheduler existed
would have invalidated every measurement taken through it — including the sweep
the real policy was later derived from. `RESERVED` is still rejected.

`PERF_MODE` reports which dataflow actually ran, so software can read back what
`ADAPTIVE` decided.

---

## 4. PE microarchitecture (`rtl/pe.sv`)

Separate architectural state, deliberately not one shared ambiguous register:

| Register | Role |
|---|---|
| `weight_reg` | stationary WS weight |
| `a_reg` / `a_valid_reg` | activation, west -> east |
| `psum_reg` / `psum_valid_reg` | partial sum, north -> south |

Weights shift north -> south during LOAD only; partial sums flow north -> south
during COMPUTE only. The two uses of the vertical direction are **temporally
disjoint**, which is what makes one shared physical array able to host a second
dataflow later instead of duplicating the mesh.

### Validity semantics (load-bearing)

```
psum_reg       <= a_valid_in ? mac_result : psum_in;
psum_valid_reg <= a_valid_in | psum_valid_in;
```

An inactive PE forwards `psum_in` untouched, so it must forward the accompanying
validity too. Regenerating validity from the local `a_valid_in` instead kills the
flag at the first inactive row — and since the array is physically `N_ARR` tall,
**no tile with `k_tile < N_ARR` would ever emit a result**. For a full tile the
two formulations are equivalent, because the activation and psum wavefronts
coincide.

### Inter-tile flush (load-bearing)

Both valid registers are **cleared whenever `phase != COMPUTE`**. Pass-through
validity means the pipeline does not self-drain by the last COMPUTE cycle, so a
valid left in flight when one tile ends would re-enter the next tile's COMPUTE
as a spurious operand. The flush makes "tiles flush before the next one begins"
an enforced property rather than an assumption — and it is what will let WS and
OS tiles interleave on the same array with no mode-switch penalty.

Data registers are not cleared: without its valid, data can never be consumed,
and leaving it saves flops.

---

## 5. Weight-stationary mapping

Array row `r` holds reduction index `kk = r`; array column `c` holds output
column `nn = c`.

- weight load: `k_tile` cycles, injected in **reverse row order** from the north
  edge, so after `k_tile` shifts `b_buf[kk]` sits in array row `kk`
- row `r` presents `A[m0+m][k0+r]` at `compute_cnt == m + r`
- column `c` emits its `m`-th result at `compute_cnt == m + c + N_ARR`
- compute length: `m_chunk + n_tile + N_ARR - 1`

The result-skew formula is **asserted in RTL**, not assumed, so a future change
that breaks the skew fails loudly rather than corrupting results silently.

### Why the drain depth is `N_ARR`, not `k_tile`

A partial sum traverses every physical row, and an inactive row still costs one
register stage. So the drain depth is `N_ARR` regardless of how short the tile
is, and the result-timing formula is unchanged by raggedness. This is why
`compute_last` uses `N_ARR` while the activation gate uses `k_tile`.

---

## 6. Tiling

Loop order:

```
for n_tile:                 // output column block
  for k_tile:               // reduction block
    fetch B tile -> weight load
    for m_chunk:            // streamed activations
      fetch A chunk -> compute -> writeback
```

The **m loop is innermost specifically so weights stay resident** across every M
chunk. Reloading per chunk would waste `k_tile` cycles per chunk and defeat the
point of weight-stationary execution. This is enforced by
`p_no_reload_within_m_loop`.

For `M=37, N=10, K=13` on a 4x4 array: K tiles `[4,4,4,1]`, N tiles `[4,4,2]`,
M chunks `[16,16,5]`.

### Ragged tiles without zero-padding

Short tiles use their **true extents**. Two independent gates produce correct
ragged behavior with no padding anywhere:

- `r < k_tile` — rows beyond the tile height never activate, so they pass psum
  through untouched. Their `weight_reg` holds stale data from a previous tile,
  which is harmless precisely because they never fire. **No zero-fill of the
  weight tile is required.**
- `0 <= m < m_chunk` — the temporal gate covering pipeline fill/drain and a
  short final M chunk.
- `c < n_tile` — columns beyond the tile width are never captured.

### C accumulation across K tiles

There is **no pre-zeroing pass**. The first K tile (`k0 == 0`) writes C
directly; every later K tile performs a read-modify-write. The RMW path is
deliberately **non-pipelined** for v1: 2 cycles per element (read half, then
write half), enforced mutually exclusive by `p_c_rd_wr_exclusive`.

The testbench poisons C with `0x5A5A5A5A` before every run, so a missed first-K
write shows up as surviving poison rather than as a silently-correct zero.

---

## 7. Control FSM

```
S_IDLE -> S_BFETCH -> S_WLOAD -> S_AFETCH -> S_COMPUTE -> S_WB -> S_NEXT
                          ^                                         |
                          |  (more k / more n)                      |
                          +-----------------------------------------+
                              S_AFETCH <-- (more m, weights resident)
S_NEXT -> S_DONE -> S_IDLE
```

### Counter convention (important)

Every counter arm in the sequential block reads the **pre-edge** state, because
`case (state)` in the same `always_ff` that does `state <= state_n` sees the old
value. Two consequences, both load-bearing:

1. A counter's arm still fires on the transition edge *out* of its own state, so
   a counter whose terminal value is its maximum representable value must
   **saturate** rather than wrap.
2. At any edge, the pre-edge register values are exactly the values that drove
   the memory bus during the cycle whose response is arriving now. This is why
   **no fetch phase needs a shadow address register**: the pre-edge coordinate
   pair *is* the in-flight address.

### Address generation

Tile base addresses (`m0*cfg_k + k0`, `k0*cfg_n + n0`, `m0*cfg_n + n0`) are
evaluated **once per tile at phase setup**, never per element. The per-element
walk is pure increment, with a row-stride correction when wrapping to the next
row of a tile.

---

## 8. Design decisions and bugs found

Recorded because the reasoning is the interesting part.

### Milestone 1

**8.1 FILL address shadow register lagged one cycle.** A shadow captured the
pre-edge counter while the bus carried the post-edge value, so every response
was written into the previous address's slot (`buf[i] = mem[i+1]`, `buf[15]`
never written). An IDLE-arm "priming" hack made address 0 coincidentally line
up, masking total corruption as subtle corruption. Predicted and confirmed
bit-exactly before fixing. **Fix:** delete the shadow entirely.

**8.2 `wb_cnt` overshoot.** Unguarded increment on the exit edge wrapped the
counter to 0 (it is exactly `CNT_W` bits) on the cycle `done` asserted.
`load_cnt` already saturated for this reason; `wb_cnt` did not.

**8.3 Weight-stability assertion sampled the wrong cycle.** `weight_reg`'s final
load happens on the same edge `phase` moves `LOAD -> COMPUTE`, so that
legitimate write first becomes visible when `phase` already reads `COMPUTE`.
**Fix:** gate on `$past(phase)` — the phase that actually enabled the write. The
invariant was corrected, not weakened.

**8.4 Unsigned-zero comparison.** `m_candidate >= '0` is constant-true, silently
defeating the activation lower-bound check. All functional tests still passed;
Verilator's `UNSIGNED` warning caught it — but only after the file-scope
`lint_off WIDTHEXPAND` was removed. That suppression had no matching `lint_on`
and would have covered all future code in the file. Replaced with width-matched
localparams; the design is now clean under `-Wall` with **no** width waiver.

### Milestone 3

**8.8 Ragged-N tiles burned MACs on discarded work.** The activation valid
travels west-to-east through every column of a row, so with `n_span < N_ARR` the
columns still holding a previous tile's weights kept multiplying. Their results
were never captured, so **every correctness test passed** while the array burned
real MACs and real power. Invisible until the MAC counter was cross-checked
against `M*N*K`: `37x10x13` reported 5763 MACs against 4810 expected, about 17%
waste. The pattern was exact — over-counting occurred precisely when
`n_span < N_ARR`. **Fix:** gate the valid entering each PE with
`column < n_active`; an inactive PE registers a zero valid, which starves the
rest of the row, so only the first inactive column needs the gate.

This is the clearest argument in the project for building the counters: a
functional testbench cannot see wasted work, only wrong work.

### Milestone 6

**8.9 ADAPTIVE skipped the accumulator clear.** The IDLE next-state decode tested
`cfg_policy == FORCE_OS` literally, so an `ADAPTIVE` run that *chose* OS entered
through the WS path and never passed through `S_OS_CLEAR`. Results were still
correct, because a completed drain shifts zeros in behind itself and leaves the
array clear by accident — so again no correctness test could see it. What caught
it was cross-checking `ADAPTIVE`'s cycle count against a forced run of the
dataflow it reported choosing: the missing clear cycles did not add up.
**Fix:** both the decode and the mode latch now key off one resolved
`start_mode_os` expression, plus an assertion (`p_os_enters_via_clear`) that an
OS transaction must enter through the clear.

### Milestone 2

**8.5 Valid propagation killed ragged-K results.** Regenerating `psum_valid`
from the local `a_valid` meant the bottom row never asserted valid when
`k_tile < N_ARR`. **Fix:** pass validity through inactive PEs alongside the data
(§4). Verified equivalent for full tiles by regression.

**8.6 `col_result_cnt` was not re-armed per tile.** Restructuring dropped the
reset, so after the first tile the counter parked at `m_chunk`, the
`< m_chunk` guard went permanently false, and `c_buf` froze at the first tile's
results — every later shape returned the *previous* test's data. Diagnosed from
exactly that signature (first test passes, all others return stale values).
**Fix:** re-arm on every entry to COMPUTE.

**8.7 Valid pipeline did not self-drain between tiles.** A consequence of 8.5:
with pass-through validity, valids are still in flight when COMPUTE ends, and
they re-entered the next tile's COMPUTE as spurious operands — caught by the
result-skew assertion, not by a data mismatch. **Fix:** explicit flush of both
valid registers whenever `phase != COMPUTE` (§4).

---

## 9. Measured results (Milestone 2)

Cycle counts from the RTL, checked against an independently-written analytical
model of the WS schedule. Every phase matches the model exactly on all 134
tested shapes.

| Shape (M x N x K) | Cycles | BFETCH | WLOAD | AFETCH | COMPUTE | WB | Array active |
|---|---|---|---|---|---|---|---|
| 4 x 4 x 4 | 65 | 16 | 4 | 16 | 11 | 16 | 16.9% |
| 4 x 4 x 8 | 145 | 32 | 8 | 32 | 22 | 48 | 15.2% |
| 5 x 5 x 5 | 207 | 25 | 10 | 50 | 42 | 75 | 20.3% |
| 17 x 4 x 4 | 190 | 16 | 4 | 68 | 31 | 68 | 16.3% |
| 4 x 4 x 20 | 385 | 80 | 20 | 80 | 55 | 144 | 14.3% |
| **37 x 10 x 13** | **4911** | 130 | 39 | 1443 | 672 | 2590 | **13.7%** |

### The headline finding

**The array is idle 80–86% of the time.** Two causes, and they point in
different architectural directions:

1. **Operand fetch is serial with compute.** `AFETCH` alone is 29% of the
   37x10x13 run. Nothing overlaps: the accelerator fetches a chunk, computes it,
   writes it back, then fetches the next. This is exactly what **double
   buffering** (Milestone 7) exists to fix, and it bounds how much that
   milestone can win.

2. **C read-modify-write dominates deep-K workloads.** `WB` is **53%** of the
   37x10x13 run — more than compute, fetch, and weight load combined. With 4 K
   tiles, three of them pay 2 cycles per output element for a non-pipelined RMW.

Finding 2 is the more interesting one, because it is a *dataflow* problem rather
than a scheduling one. Weight-stationary execution must spill and re-read
partial sums through C whenever K exceeds the array height. **Output-stationary
execution keeps the accumulator resident in the PE and pays none of that.**

That was the concrete, measured hypothesis for Milestones 4–5 to test:

> WS should lose to OS as K grows relative to `N_ARR`, and the crossover should
> track the point where accumulated C read-modify-write traffic exceeds OS's
> fixed drain cost.

**The hypothesis was confirmed, with a correction.** The crossover does track K,
but it is not fixed — it *rises with M*, because OS's own overheads (re-reading
B once per M tile, and a clear plus drain per output tile) also scale with the
number of M tiles. The measurements are in §9b.

### Model vs. frozen cost model

The frozen cost model was

```
WS_cost(tile) = M_tile + N_tile + 2*K_tile - 1 + stall_WS + accum_rmw_cost * (...)
```

The measured per-tile architectural cost is `k_tile` (load) plus
`m_chunk + n_tile + N_ARR - 1` (compute). These agree **only when
`k_tile == N_ARR`**; for a ragged K tile the model's `2*K_tile` term
underestimates, because the psum drain depth stays at `N_ARR` even when the tile
is short. Worth carrying into the Milestone 5 comparison rather than quietly
correcting.

---

## 9b. WS vs OS measurement (Milestone 5)

Both dataflows over a 56-shape grid (`M in {1,4,16,64}`, `N in {4,16}`,
`K in {1,2,4,8,16,32,64}`), identical data, each run verified against the
golden model. Full data in `docs/benchmark.csv`; regenerate with `make bench`.

**WS wins 14, OS wins 30, 12 ties.**

| M | WS wins | OS wins | crossover |
|---|---|---|---|
| 1, 4 | — (ties for small K) | K >= 8 | at K = N_ARR |
| 16 | K <= 4 | K >= 8 | K = 8 |
| 64 | K <= 8 | K >= 16 | K = 16 |

Representative points:

| Shape | WS cycles | OS cycles | OS gain |
|---|---|---|---|
| 4 x 4 x 4 | 65 | 65 | tie |
| 4 x 4 x 64 | 1265 | 623 | +50.8% |
| 64 x 4 x 4 | 629 | 1025 | -63.0% |
| 64 x 16 x 8 | 6049 | 6401 | -5.8% |
| 64 x 16 x 64 | 55553 | 39809 | +28.3% |

The mechanism is visible directly in the phase breakdown. On `37x10x13`,
writeback collapses `2590 -> 370` cycles while B fetch rises `130 -> 1300`:
OS trades C traffic for B traffic. A-operand traffic is **identical** in both
dataflows and cancels exactly, which is what makes the comparison tractable.

### Why the crossover moves with M

- WS pays `2*M*N*(Ktiles-1)` extra C accesses — grows with K.
- OS pays `K*N*(Mtiles-1)` extra B reads and `(N_ARR+1)*Mtiles*Ntiles` of
  clear/drain — both grow with M.

So larger M pushes the crossover to larger K. A rule keyed on K alone cannot
capture this, which the policy scoring below confirms.

---

## 9c. Choosing a policy (Milestone 6)

Candidates scored against the *measured* winner, not against the analytical
model. Regret is cycles lost versus an oracle that always picks correctly.

| Policy | Correct | Total regret | Worst case |
|---|---|---|---|
| always WS | 26/56 | 48190 | 103.2% |
| always OS | 42/56 | 5390 | 63.0% |
| `K > N_ARR` | 54/56 | 440 | 5.8% |
| cost-difference model | **56/56** | **0** | **0.0%** |
| ADAPTIVE (in hardware) | **56/56** | **0** | **0.0%** |

`K > N_ARR` is the intuitive rule and it is nearly right, but it misses the
large-M, `K=8` corner where OS's per-M-tile overheads still outweigh the
writeback it saves.

The shipped rule compares the two cost differences directly:

```
gain = 2*M*N*(Ktiles-1) + Ntiles*K
cost = K*N*(Mtiles-1) + (N_ARR+1)*Mtiles*Ntiles
choose OS when gain > cost
```

Evaluated once per transaction at start (not per cycle), so its multipliers are
a one-shot cost off the critical path. The tile counts are ceiling-divides by
`N_ARR`, which is a shift because `N_ARR` is a power of two.

**Scope of the claim.** This is a measured fit to *this* memory model — fixed
1-cycle latency, no contention. It is not a universal law, and the honest way
to extend it is to re-run `make bench` after any change that alters the cost
balance. That warning was written before Milestone 7, and §9d is what happened
when it was tested.

---

## 9d. Double buffering, and what it did to the policy (Milestone 7)

### The prefetch engine

The A fetch is no longer an FSM phase. It is an engine launched by whichever
state knows the next chunk's address, which then walks the tile on its own in
whatever state the FSM happens to be in. Two consequences, both impossible
while the fetch was a state:

- an operand fetch overlaps a compute and a writeback;
- the A and B buses run concurrently, which matters in OS, where the next K
  chunk's A fetch now overlaps that chunk's B fetch.

`a_buf` gains a bank dimension. The engine fills one bank while the array
computes from the other; they swap when a completed fetch is consumed, so a
prefetch can never write the bank under compute. The swap happens on the *same*
edge that captures the last element — both are non-blocking, so the bank becomes
readable exactly as its final write lands.

`S_AFETCH` leaves as soon as the fetch has landed **or lands on that edge**. An
earlier version waited a cycle to observe completion, which taxed every
cold-start fetch; that made OS 30 cycles slower on `37x10x13` while WS got
faster, and the asymmetry is what pointed at the cold-start path.

The prefetch targets the next iteration of whichever loop is innermost: the next
M chunk in WS, the next K chunk in OS. The in-flight fetch keeps its own
snapshot of the spans it was launched with, because the live spans move with the
loop origins and those advance before the prefetch is consumed.

| Workload | Before | After | Gain |
|---|---|---|---|
| WS `37x10x13` | 4911 | 4126 | 16.0% |
| WS visible A fetch | 1443 | 658 | 54% hidden |
| OS `4x4x20` | 215 | 200 | 7.0% |
| OS `37x10x13` | 3835 | 3835 | none |

OS gains least, and the reason is structural: its inner loop is K, so when
`K <= STREAM_DEPTH` there is only one chunk per output tile and nothing to
prefetch at all.

### The policy broke

Re-running the sweep, `ADAPTIVE` had fallen from **56/56 with zero regret** to
**54/56 with a 9.0% worst case**. WS wins rose from 14 to 16.

Making the hardware faster invalidated the scheduler. The original model let
A-operand traffic cancel between the dataflows, because both moved the same
elements at the same cost. Once A can overlap other work that stops holding,
because the two dataflows have very different windows to hide it in:

```
WS inner loop is M -> A hides behind compute AND writeback; only the first
   chunk of each tile pays in full   ->  ~Ntiles * min(M,SD) * K visible
OS inner loop is K -> A hides only behind the tail of one compute plus the
   next B fetch, and often not at all ->  ~Ntiles * M * min(K,SD) visible
```

Adding that difference restores 56/56 with zero regret.

| Policy | Before prefetch | After prefetch |
|---|---|---|
| always WS | 26/56, worst 103.2% | 28/56, worst 191.8% |
| always OS | 42/56, worst 63.0% | 40/56, worst 133.2% |
| `K > N_ARR` | 54/56, worst 5.8% | 52/56, worst **41.1%** |
| cost model v1 | **56/56, worst 0%** | 54/56, worst 9.0% |
| cost model + prefetch | — | **56/56, worst 0%** |

Every simpler rule degraded, and more severely than the cost model did:
`K > N_ARR` went from a 5.8% worst case to 41.1%.

**The lesson worth keeping.** A heuristic fitted to one microarchitecture does
not survive a change to that microarchitecture. Re-running the sweep is part of
changing the hardware, not an optional follow-up — and the reason this project
could notice at all is that the policy was scored against measured cycles rather
than assumed correct.

---

## 10. Assertion inventory

| Assertion | Location | Invariant |
|---|---|---|
| `p_weight_stable_outside_load` | `pe.sv` | `weight_reg` changes only under LOAD |
| row-skew check | `pe_array.sv` | an active PE receives a valid psum from the north |
| `g_capture` timing | `accel_top.sv` | results emerge at `compute_cnt == m + c + N_ARR` |
| `p_accum_stable_without_mac` | `pe.sv` | OS accumulator changes only on a real MAC |
| `p_c_rd_wr_exclusive` | `accel_top.sv` | C read and write never in the same cycle |
| `p_no_rmw_on_first_k` | `accel_top.sv` | direct-write tiles never read C back |
| `p_os_never_reads_c` | `accel_top.sv` | OS never reads C: no accumulator escapes |
| `p_os_enters_via_clear` | `accel_top.sv` | OS transactions start from a cleared array |
| `p_mode_stable_while_busy` | `accel_top.sv` | dataflow fixed for a transaction |
| `p_no_reload_within_m_loop` | `accel_top.sv` | weights stay resident across M chunks |
| `p_tile_extents_legal` | `accel_top.sv` | spans non-zero and within STREAM_DEPTH |
| `p_spatial_span_fits` | `accel_top.sv` | spatially-mapped span fits the array |
| `p_macs_bounded` | `accel_top.sv` | never more concurrent MACs than PEs |
| `p_no_macs_outside_compute` | `accel_top.sv` | no MACs reported outside a compute phase |
| `p_busy_not_with_done` | `accel_top.sv` | `busy` and `done` never overlap |

Not decorative: six have fired during development and five caught real RTL
bugs (8.3, 8.6 via its data signature, 8.7, 8.8 via the MAC cross-check, 8.9).

The row-skew check is stated as an **implication**, not an equality: an active
PE must be receiving a valid psum from the north. The converse fails legitimately
for ragged tiles, where inactive rows forward a valid psum while not firing.

---

## 11. Verification

Software golden GEMM: signed INT8 in, 32-bit signed accumulation, no saturation.

- **Value coverage** (4x4x4): identity, all-negative, zeros, mixed signs, INT8
  `127`/`-128` extremes
- **Shape coverage**: 14 directed shapes including N-only tiling, K-only tiling
  (RMW), both, ragged N, ragged K, ragged in all three dimensions, `1x1x1`,
  thin row/column, M chunking at and past `STREAM_DEPTH`, deep K (5 tiles), and
  the spec's own `37x10x13`
- **120 randomized shapes** with random INT8 data, fixed seed
- **Control**: reset mid-transaction with clean recovery, back-to-back runs with
  no intervening reset, all four policy encodings, six illegal shapes
- **Per-run invariants**: every phase's cycle count vs. the analytical model,
  exact C write count (`M*N*ceil(K/N_ARR)`), no surviving poison, and every
  memory access bounds-checked by the harness

---

## 12. Future work

All seven planned milestones are delivered: output-stationary with its
east-flowing drain path (§4), hardware performance counters (§9), the
measurement sweep (§9b), the shape-aware `ADAPTIVE` policy (§9c), and
double-buffered prefetch with the policy re-derivation it forced (§9d).

What remains, in rough order of value:

- **Scale to 8x8 and re-run the sweep.** The crossover should move again, and
  §9d is the template for handling it.
- **Prefetch across tile boundaries.** Today the prefetch covers only the
  innermost loop, so a tile's first chunk is never hidden. OS benefits least
  precisely because its inner loop often runs exactly once — cross-tile
  prefetch is the fix, and it would shift the crossover back toward OS.
- **Pipeline the C read-modify-write path**, deliberately left non-pipelined
  in v1. It is still the single largest phase for deep-K WS workloads.
- **Per-tile adaptive selection.** The decision is currently per-transaction;
  the shared control path already supports switching, and the PE flush makes
  mode changes safe between tiles.
