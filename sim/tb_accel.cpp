// Milestone 2 testbench: tiled weight-stationary GEMM.
//
// Covers arbitrary M x N x K including ragged tiles and multi-K-tile
// accumulation, checked against a software golden model and against an
// analytical cycle model derived from the WS schedule.
#include "Vaccel_top.h"
#include "Vaccel_top___024root.h"  // public_flat_rd access to the FSM state
#include "verilated.h"
#include "verilated_vcd_c.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

static const int N_ARR        = 4;
static const int STREAM_DEPTH = 16;

static vluint64_t g_time = 0;
double sc_time_stamp() { return static_cast<double>(g_time); }

// Mirrors state_e in accel_top.sv.
enum State {
  S_IDLE = 0, S_BFETCH = 1, S_WLOAD = 2, S_AFETCH = 3,
  S_COMPUTE = 4, S_WB = 5, S_NEXT = 6, S_DONE = 7,
  S_OS_CLEAR = 8, S_OS_COMPUTE = 9, S_OS_DRAIN = 10
};

enum Policy { FORCE_WS = 0, FORCE_OS = 1, ADAPTIVE = 2, RESERVED = 3 };

// C is poisoned before every run. The design performs no pre-zeroing
// pass, so the first K tile must overwrite this value outright; if any
// output still reads as poison, a C write was missed.
static const int32_t C_POISON = 0x5A5A5A5A;

// Performance counter map; mirrors the PERF_* localparams in accel_top.
enum Perf {
  PERF_TOTAL_CYCLES = 0, PERF_COMPUTE_CYCLES = 1, PERF_BFETCH = 2,
  PERF_WLOAD = 3, PERF_AFETCH = 4, PERF_WB = 5, PERF_STALL = 6,
  PERF_WEIGHT_TILES = 7, PERF_M_CHUNKS = 8, PERF_MAC_OPS = 9,
  PERF_MAC_SLOTS = 10, PERF_BYTES_A = 11, PERF_BYTES_B = 12,
  PERF_C_WRITES = 13, PERF_C_RMW = 14, PERF_ERRORS = 15,
  PERF_OS_CLEAR = 16, PERF_OS_DRAIN = 17, PERF_MODE = 18
};

struct PhaseCounts {
  int bfetch = 0, wload = 0, afetch = 0, compute = 0, wb = 0, next = 0, done = 0;
  int os_clear = 0, os_drain = 0, other = 0;
  int total() const {
    return bfetch + wload + afetch + compute + wb + next + done +
           os_clear + os_drain + other;
  }
  // Cycles the systolic array is actually computing, vs. everything else.
  int useful() const { return compute; }
};

// Analytical models of both schedules, written independently of the RTL
// so that a disagreement is informative rather than circular.
//
// The structural difference that matters: WS tiles K onto the array and
// streams M, so a K larger than the array forces partial sums out to
// memory and back (the 2x writeback below). OS tiles M onto the array
// and streams K, so the reduction completes in place and C is written
// exactly once, at the cost of re-fetching B for every M tile.
// S_AFETCH is a wait state under double buffering: it costs one cycle to
// observe the fetch landed and swap banks, plus however much of the
// fetch was left unhidden. A cold start (first chunk of a tile) has no
// overlap window at all; later chunks were prefetched during the
// previous compute, so only the excess over that window is visible.
static int afetch_cost(int fetch, int overlap) {
  if (overlap < 0) return fetch;                  // cold start: no prefetch
  return std::max(1, fetch - overlap);            // 1 cycle if fully hidden
}

static PhaseCounts model_ws(int M, int Nn, int K) {
  PhaseCounts p;
  for (int n0 = 0; n0 < Nn; n0 += N_ARR) {
    int n_tile = std::min(N_ARR, Nn - n0);
    for (int k0 = 0; k0 < K; k0 += N_ARR) {
      int k_tile = std::min(N_ARR, K - k0);
      bool k_first = (k0 == 0);
      p.bfetch += k_tile * n_tile;
      p.wload  += k_tile;
      int overlap = -1;  // no prefetch window for the first M chunk
      for (int m0 = 0; m0 < M; m0 += STREAM_DEPTH) {
        int m_chunk = std::min(STREAM_DEPTH, M - m0);
        int compute_len = m_chunk + n_tile + N_ARR - 1;
        int wb_len      = m_chunk * n_tile * (k_first ? 1 : 2);
        p.afetch  += afetch_cost(m_chunk * k_tile, overlap);
        p.compute += compute_len;
        p.wb      += wb_len;
        p.next    += 1;
        // The next chunk's fetch is launched one cycle into this
        // compute and runs through the rest of it, the writeback and
        // the loop-advance cycle.
        overlap = compute_len + wb_len;
      }
    }
  }
  p.done = 1;
  return p;
}

static PhaseCounts model_os(int M, int Nn, int K) {
  PhaseCounts p;
  for (int m0 = 0; m0 < M; m0 += N_ARR) {
    int m_tile = std::min(N_ARR, M - m0);
    for (int n0 = 0; n0 < Nn; n0 += N_ARR) {
      int n_tile = std::min(N_ARR, Nn - n0);
      p.os_clear += 1;
      int overlap = -1;  // no prefetch window for the first K chunk
      for (int k0 = 0; k0 < K; k0 += STREAM_DEPTH) {
        int k_chunk = std::min(STREAM_DEPTH, K - k0);
        int compute_len = k_chunk + m_tile + n_tile - 2;
        int bfetch_len  = k_chunk * n_tile;
        p.bfetch  += bfetch_len;
        p.afetch  += afetch_cost(m_tile * k_chunk, overlap);
        p.compute += compute_len;
        // The next K chunk's A fetch is launched one cycle into this
        // compute and keeps running through that chunk's B fetch --
        // the two memory interfaces are independent.
        int next_k = std::min(STREAM_DEPTH, K - (k0 + k_chunk));
        overlap = (compute_len - 1) + (next_k > 0 ? next_k * n_tile : 0);
      }
      p.os_drain += N_ARR;   // chain is N_ARR wide regardless of n_tile
      p.wb       += m_tile * n_tile;  // always a direct write
      p.next     += 1;
    }
  }
  p.done = 1;
  return p;
}

static PhaseCounts model_cycles(int M, int Nn, int K, int policy) {
  return (policy == 1) ? model_os(M, Nn, K) : model_ws(M, Nn, K);
}

static int g_failures = 0;

static void check(bool cond, const std::string& msg) {
  if (!cond) {
    std::printf("  [FAIL] %s\n", msg.c_str());
    g_failures++;
  }
}

struct SimMem {
  std::vector<int8_t>  a, b;
  std::vector<int32_t> c;
  bool a_pend = false, b_pend = false, c_pend = false;
  uint32_t a_paddr = 0, b_paddr = 0, c_paddr = 0;
};

class Harness {
 public:
  Vaccel_top* dut;
  VerilatedVcdC* tfp;
  SimMem mem;
  int c_writes = 0;
  bool trace_en = true;

  Harness(Vaccel_top* d, VerilatedVcdC* t) : dut(d), tfp(t) {}

  void tick() {
    // Serve the requests captured last cycle (fixed 1-cycle latency).
    dut->mem_a_rvalid = mem.a_pend;
    dut->mem_a_rdata  = mem.a_pend ? mem.a[mem.a_paddr] : 0;
    dut->mem_b_rvalid = mem.b_pend;
    dut->mem_b_rdata  = mem.b_pend ? mem.b[mem.b_paddr] : 0;
    dut->mem_c_rvalid = mem.c_pend;
    dut->mem_c_rdata  = mem.c_pend ? mem.c[mem.c_paddr] : 0;

    dut->clk = 0;
    dut->eval();
    if (tfp && trace_en) tfp->dump(g_time++); else g_time++;

    dut->clk = 1;
    dut->eval();
    if (tfp && trace_en) tfp->dump(g_time++); else g_time++;

    if (dut->mem_a_rd_en) { check(dut->mem_a_addr < mem.a.size(), "A read out of range"); }
    if (dut->mem_b_rd_en) { check(dut->mem_b_addr < mem.b.size(), "B read out of range"); }
    if (dut->mem_c_rd_en || dut->mem_c_wr_en) {
      check(dut->mem_c_addr < mem.c.size(), "C access out of range");
    }

    mem.a_pend = dut->mem_a_rd_en; mem.a_paddr = dut->mem_a_addr;
    mem.b_pend = dut->mem_b_rd_en; mem.b_paddr = dut->mem_b_addr;
    mem.c_pend = dut->mem_c_rd_en; mem.c_paddr = dut->mem_c_addr;

    if (dut->mem_c_wr_en) {
      mem.c[dut->mem_c_addr] = dut->mem_c_wdata;
      c_writes++;
    }
  }

  void reset(int cycles = 4) {
    dut->rst_n = 0;
    dut->start = 0;
    dut->cfg_m = 0; dut->cfg_n = 0; dut->cfg_k = 0; dut->cfg_policy = 0;
    mem.a_pend = mem.b_pend = mem.c_pend = false;
    for (int i = 0; i < cycles; i++) tick();
    dut->rst_n = 1;
    tick();
  }

  int cur_state() const { return dut->rootp->accel_top__DOT__state; }

  // Combinational counter read: drive the address, settle, sample.
  uint32_t perf(int idx) {
    dut->perf_addr = idx;
    dut->eval();
    return dut->perf_rdata;
  }

  void account(PhaseCounts& p) {
    switch (cur_state()) {
      case S_BFETCH:  p.bfetch++;  break;
      case S_WLOAD:   p.wload++;   break;
      case S_AFETCH:  p.afetch++;  break;
      case S_COMPUTE: p.compute++; break;
      case S_WB:      p.wb++;      break;
      case S_NEXT:    p.next++;    break;
      case S_DONE:    p.done++;    break;
      case S_OS_CLEAR:   p.os_clear++; break;
      case S_OS_COMPUTE: p.compute++;  break;
      case S_OS_DRAIN:   p.os_drain++; break;
      default:        p.other++;   break;
    }
  }

  PhaseCounts run_gemm(int M, int Nn, int K,
                       const std::vector<int8_t>& A, const std::vector<int8_t>& B,
                       int policy = FORCE_WS, int max_cycles = 2000000) {
    mem.a = A;
    mem.b = B;
    mem.c.assign(static_cast<size_t>(M) * Nn, C_POISON);
    c_writes = 0;

    dut->cfg_m = M; dut->cfg_n = Nn; dut->cfg_k = K;
    dut->cfg_policy = policy;
    dut->start = 1;
    tick();
    dut->start = 0;

    PhaseCounts p;
    account(p);
    int cycles = 1;
    while (!dut->done && cycles < max_cycles) {
      tick();
      cycles++;
      account(p);
    }
    check(dut->done, "run did not complete (done never asserted)");
    tick();
    return p;
  }
};

static void golden(int M, int Nn, int K, const std::vector<int8_t>& A,
                   const std::vector<int8_t>& B, std::vector<int32_t>& C) {
  C.assign(static_cast<size_t>(M) * Nn, 0);
  for (int m = 0; m < M; m++)
    for (int n = 0; n < Nn; n++) {
      int32_t acc = 0;
      for (int k = 0; k < K; k++)
        acc += int32_t(A[m * K + k]) * int32_t(B[k * Nn + n]);
      C[m * Nn + n] = acc;
    }
}

// Runs one shape and checks: golden correctness, no surviving poison,
// exact C write count, and measured-vs-modelled cycles per phase.
static PhaseCounts shape_test(Harness& h, const std::string& name, int M, int Nn, int K,
                       const std::vector<int8_t>& A, const std::vector<int8_t>& B,
                       bool verbose = true, int policy = FORCE_WS) {
  int before = g_failures;
  PhaseCounts got = h.run_gemm(M, Nn, K, A, B, policy);
  std::vector<int32_t> gold;
  golden(M, Nn, K, A, B, gold);

  int bad = 0;
  for (int m = 0; m < M && bad < 5; m++)
    for (int n = 0; n < Nn && bad < 5; n++) {
      int32_t g = gold[m * Nn + n], r = h.mem.c[m * Nn + n];
      if (r != g) {
        std::printf("  [FAIL] %s: C[%d][%d] = %d, expected %d\n", name.c_str(), m, n, r, g);
        g_failures++; bad++;
      }
    }
  for (size_t i = 0; i < h.mem.c.size(); i++) {
    if (h.mem.c[i] == C_POISON && gold[i] != C_POISON) {
      check(false, name + ": output element never written (poison survived)");
      break;
    }
  }

  // WS writes each output once per K tile (once directly, then once per
  // read-modify-write pass). OS writes each output exactly once, because
  // the reduction never leaves the array -- this is the core structural
  // difference between the two dataflows.
  bool os = (policy == FORCE_OS);
  int k_tiles = (K + N_ARR - 1) / N_ARR;
  int expect_writes = os ? (M * Nn) : (M * Nn * k_tiles);
  check(h.c_writes == expect_writes,
        name + ": C writes = " + std::to_string(h.c_writes) + ", expected " +
            std::to_string(expect_writes));

  PhaseCounts exp = model_cycles(M, Nn, K, policy);
  check(got.os_clear == exp.os_clear, name + ": OS_CLEAR " + std::to_string(got.os_clear) + " vs model " + std::to_string(exp.os_clear));
  check(got.os_drain == exp.os_drain, name + ": OS_DRAIN " + std::to_string(got.os_drain) + " vs model " + std::to_string(exp.os_drain));
  check(got.bfetch  == exp.bfetch,  name + ": BFETCH "  + std::to_string(got.bfetch)  + " vs model " + std::to_string(exp.bfetch));
  check(got.wload   == exp.wload,   name + ": WLOAD "   + std::to_string(got.wload)   + " vs model " + std::to_string(exp.wload));
  check(got.afetch  == exp.afetch,  name + ": AFETCH "  + std::to_string(got.afetch)  + " vs model " + std::to_string(exp.afetch));
  check(got.compute == exp.compute, name + ": COMPUTE " + std::to_string(got.compute) + " vs model " + std::to_string(exp.compute));
  check(got.wb      == exp.wb,      name + ": WB "      + std::to_string(got.wb)      + " vs model " + std::to_string(exp.wb));
  check(got.next    == exp.next,    name + ": NEXT "    + std::to_string(got.next)    + " vs model " + std::to_string(exp.next));
  check(got.other   == 0,           name + ": cycles outside defined phases");

  // ---- Hardware performance counters ----
  //
  // The headline invariant: the array must perform exactly M*N*K
  // multiply-accumulates. One MAC per (m,n,k) triple, no more and no
  // fewer. This is measured from the array's own valid popcount, so it
  // independently proves both that no work is missing and that ragged
  // tiles waste nothing -- a zero-padding implementation would report
  // more MACs than M*N*K.
  uint32_t mac_ops  = h.perf(PERF_MAC_OPS);
  uint32_t expect_macs = static_cast<uint32_t>(M) * Nn * K;
  check(mac_ops == expect_macs,
        name + ": MAC ops = " + std::to_string(mac_ops) + ", expected M*N*K = " +
            std::to_string(expect_macs));

  // Counters must agree with the externally observed phase occupancy.
  check(h.perf(PERF_COMPUTE_CYCLES) == static_cast<uint32_t>(got.compute),
        name + ": compute-cycle counter disagrees with observed occupancy");
  check(h.perf(PERF_BFETCH) == static_cast<uint32_t>(got.bfetch),
        name + ": bfetch counter disagrees");
  check(h.perf(PERF_AFETCH) == static_cast<uint32_t>(got.afetch),
        name + ": afetch counter disagrees");
  check(h.perf(PERF_WB) == static_cast<uint32_t>(got.wb),
        name + ": wb counter disagrees");
  check(h.perf(PERF_TOTAL_CYCLES) == static_cast<uint32_t>(got.total() - got.done),
        name + ": total-cycle counter disagrees with busy occupancy");
  check(h.perf(PERF_C_WRITES) == static_cast<uint32_t>(expect_writes),
        name + ": C-write counter disagrees with observed writes");
  // OS must never read C back at all: the accumulator stays in the PE
  // for the entire reduction.
  check(h.perf(PERF_C_RMW) ==
            (os ? 0u : static_cast<uint32_t>(M) * Nn * (k_tiles - 1)),
        name + ": C read-modify-write counter wrong");
  if (!os) {
    check(h.perf(PERF_WEIGHT_TILES) ==
              static_cast<uint32_t>(((Nn + N_ARR - 1) / N_ARR) * k_tiles),
          name + ": weight-tile counter wrong");
  }
  // Operand traffic, counted as elements moved rather than as cycles
  // spent -- under double buffering the A fetch overlaps other phases,
  // so its cycle count and its byte count are no longer the same
  // number. Both dataflows re-read the full A matrix once per N tile.
  int n_tiles = (Nn + N_ARR - 1) / N_ARR;
  check(h.perf(PERF_BYTES_B) == static_cast<uint32_t>(exp.bfetch),
        name + ": B byte counter disagrees with fetch cycles");
  check(h.perf(PERF_BYTES_A) == static_cast<uint32_t>(M) * K * n_tiles,
        name + ": A byte counter = " + std::to_string(h.perf(PERF_BYTES_A)) +
            ", expected M*K*Ntiles = " + std::to_string(M * K * n_tiles));

  if (verbose) {
    uint32_t slots = h.perf(PERF_MAC_SLOTS);
    double occ_compute = slots ? 100.0 * mac_ops / slots : 0.0;
    double occ_overall = 100.0 * mac_ops /
                         (static_cast<double>(got.total()) * N_ARR * N_ARR);
    std::printf("  %-16s %3dx%3dx%3d %s %s cyc=%5d [bf %4d wl %3d af %4d cp %4d wb %4d]"
                " MACs=%6u occ %5.1f%%/%4.1f%%\n",
                name.c_str(), M, Nn, K, os ? "OS" : "WS",
                g_failures == before ? "PASS" : "FAIL",
                got.total(), got.bfetch, got.wload, got.afetch, got.compute,
                got.wb, mac_ops, occ_compute, occ_overall);
  }
  return got;
}

static uint32_t g_rng = 0xC0FFEEu;
static uint32_t rnd() {
  g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5;
  return g_rng;
}

static void fill_random(std::vector<int8_t>& v, size_t n) {
  v.resize(n);
  for (size_t i = 0; i < n; i++) v[i] = static_cast<int8_t>(rnd() & 0xFF);
}

// ---------------------------------------------------------------------
// Milestone 5: WS vs OS measurement sweep.
//
// Runs both dataflows over a grid of shapes on identical data, verifies
// each against the golden model, and records measured cycles. The point
// is to locate the crossover empirically rather than to assume it -- the
// adaptive policy in the next milestone is derived from this table, not
// from the analytical cost model.
// ---------------------------------------------------------------------
static int ceil_div(int a, int b) { return (a + b - 1) / b; }

// ---- Candidate scheduling policies ----
// Each returns true for "choose OS". They are evaluated against the
// measured winner below, so the policy that ships is the one the data
// selects rather than the one that sounds right.

// The obvious rule: OS pays off once the reduction outgrows the array,
// because that is when WS starts spilling partial sums through C.
static bool policy_k_gt_narr(int M, int Nn, int K) {
  (void)M; (void)Nn;
  return K > N_ARR;
}

// Cost-difference rule, derived from the structural asymmetry between
// the two dataflows rather than fitted to the measurements:
//
//   WS pays 2 extra C accesses per output for every K tile after the
//   first            -> 2*M*N*(Ktiles-1)
//   OS re-reads the whole B tile once per M tile
//                    -> K*N*(Mtiles-1)
//   OS pays a clear and an N_ARR-cycle drain per output tile
//                    -> (N_ARR+1)*Mtiles*Ntiles
//   WS pays a weight load per (n,k) tile
//                    -> Ntiles*K
//
// A-operand traffic is identical in both and cancels exactly.
static bool policy_cost_model(int M, int Nn, int K) {
  int ktiles = ceil_div(K, N_ARR);
  int mtiles = ceil_div(M, N_ARR);
  int ntiles = ceil_div(Nn, N_ARR);
  long os_gain = 2L * M * Nn * (ktiles - 1) + (long)ntiles * K;
  long os_cost = (long)K * Nn * (mtiles - 1) + (long)(N_ARR + 1) * mtiles * ntiles;
  return os_gain > os_cost;
}

// Same idea, corrected for double buffering.
//
// The original model let A-operand traffic cancel, because both
// dataflows moved the same A elements at the same cost. With a prefetch
// engine that is no longer true: each dataflow hides A behind whatever
// its inner loop leaves running, and those windows are very different.
//
//   WS's inner loop is M, so its A fetch hides behind compute AND the
//      writeback -- a long window, and only the first chunk of each
//      tile pays in full: ~Ntiles * min(M,STREAM_DEPTH) * K visible.
//   OS's inner loop is K, so its A fetch hides only behind the rest of
//      one compute plus the next B fetch, and when K <= STREAM_DEPTH
//      there is no second chunk to prefetch at all:
//      ~Ntiles * M * min(K,STREAM_DEPTH) visible.
//
// The difference of those two is what the corrected rule adds.
static bool policy_cost_model_db(int M, int Nn, int K) {
  int ktiles = ceil_div(K, N_ARR);
  int mtiles = ceil_div(M, N_ARR);
  int ntiles = ceil_div(Nn, N_ARR);
  int ws_vis_a = ntiles * std::min(M, STREAM_DEPTH) * K;
  int os_vis_a = ntiles * M * std::min(K, STREAM_DEPTH);
  long os_gain = 2L * M * Nn * (ktiles - 1) + (long)ntiles * K
               + ((long)ws_vis_a - (long)os_vis_a);
  long os_cost = (long)K * Nn * (mtiles - 1) + (long)(N_ARR + 1) * mtiles * ntiles;
  return os_gain > os_cost;
}

struct PolicyStat {
  const char* name;
  int correct = 0;
  long regret = 0;      // total cycles lost vs always picking the winner
  double worst = 0.0;   // worst single-shape loss, percent
};

static void score(PolicyStat& s, bool pick_os, int wc, int oc) {
  int chosen = pick_os ? oc : wc;
  int best   = std::min(wc, oc);
  if (chosen == best) s.correct++;
  s.regret += (chosen - best);
  double loss = 100.0 * (chosen - best) / static_cast<double>(best);
  if (loss > s.worst) s.worst = loss;
}

static void run_sweep(Harness& h) {
  h.trace_en = false;

  std::FILE* csv = std::fopen("../docs/benchmark.csv", "w");
  if (csv) {
    std::fprintf(csv, "M,N,K,ws_cycles,os_cycles,winner,os_speedup,"
                      "ws_wb,os_wb,ws_bfetch,os_bfetch,ws_afetch,os_afetch,"
                      "ws_compute,os_compute,macs\n");
  } else {
    std::printf("note: could not open ../docs/benchmark.csv for writing; "
                "results are printed below only\n");
  }

  const int Ms[] = {1, 4, 16, 64};
  const int Ns[] = {4, 16};
  const int Ks[] = {1, 2, 4, 8, 16, 32, 64};

  std::printf("\n%6s %4s %4s | %8s %8s | %-6s %7s | %s\n",
              "M", "N", "K", "WS cyc", "OS cyc", "winner", "OS gain", "where the cycles go");
  std::printf("%s\n", std::string(96, '-').c_str());

  int ws_wins = 0, os_wins = 0, ties = 0, points = 0;
  PolicyStat st_ws{"always WS"}, st_os{"always OS"};
  PolicyStat st_k{"K > N_ARR"}, st_cm{"cost-difference model"};
  PolicyStat st_db{"cost model + prefetch"};
  PolicyStat st_hw{"ADAPTIVE (in hardware)"};

  for (int mi = 0; mi < 4; mi++)
    for (int ni = 0; ni < 2; ni++)
      for (int ki = 0; ki < 7; ki++) {
        int M = Ms[mi], Nn = Ns[ni], K = Ks[ki];
        std::vector<int8_t> A, B;
        fill_random(A, static_cast<size_t>(M) * K);
        fill_random(B, static_cast<size_t>(K) * Nn);
        std::vector<int32_t> gold;
        golden(M, Nn, K, A, B, gold);

        PhaseCounts ws = h.run_gemm(M, Nn, K, A, B, FORCE_WS);
        bool ws_ok = (h.mem.c == gold);
        uint32_t macs = h.perf(PERF_MAC_OPS);

        PhaseCounts os = h.run_gemm(M, Nn, K, A, B, FORCE_OS);
        bool os_ok = (h.mem.c == gold);

        check(ws_ok, "sweep: WS result wrong");
        check(os_ok, "sweep: OS result wrong");

        int wc = ws.total(), oc = os.total();
        const char* win = (oc < wc) ? "OS" : (wc < oc) ? "WS" : "tie";
        if (oc < wc) os_wins++; else if (wc < oc) ws_wins++; else ties++;
        double gain = 100.0 * (wc - oc) / static_cast<double>(wc);

        // ADAPTIVE on the same shape: the hardware picks for itself, and
        // must land on the faster of the two and still be correct.
        PhaseCounts ad = h.run_gemm(M, Nn, K, A, B, ADAPTIVE);
        check(h.mem.c == gold, "sweep: ADAPTIVE result wrong");
        bool ad_chose_os = (h.perf(PERF_MODE) != 0);
        int ac = ad.total();
        check(ac == (ad_chose_os ? oc : wc),
              "sweep: ADAPTIVE cycles do not match the dataflow it reported");

        points++;
        score(st_ws, false, wc, oc);
        score(st_os, true,  wc, oc);
        score(st_k,  policy_k_gt_narr(M, Nn, K), wc, oc);
        score(st_cm, policy_cost_model(M, Nn, K), wc, oc);
        score(st_db, policy_cost_model_db(M, Nn, K), wc, oc);
        score(st_hw, ad_chose_os, wc, oc);

        std::printf("%6d %4d %4d | %8d %8d | %-6s %6.1f%% | "
                    "wb %5d->%-5d  bf %5d->%-5d\n",
                    M, Nn, K, wc, oc, win, gain, ws.wb, os.wb, ws.bfetch, os.bfetch);

        if (csv) {
          std::fprintf(csv, "%d,%d,%d,%d,%d,%s,%.4f,%d,%d,%d,%d,%d,%d,%d,%d,%u\n",
                       M, Nn, K, wc, oc, win, gain / 100.0,
                       ws.wb, os.wb, ws.bfetch, os.bfetch,
                       ws.afetch, os.afetch, ws.compute, os.compute, macs);
        }
      }

  if (csv) std::fclose(csv);
  std::printf("%s\n", std::string(96, '-').c_str());
  std::printf("WS wins: %d   OS wins: %d   ties: %d   (of %d shapes)\n",
              ws_wins, os_wins, ties, points);

  // Which scheduling rule should the adaptive policy use? Scored against
  // the measured winner, not against the analytical model.
  std::printf("\nPolicy evaluation (oracle = always pick the measured winner)\n");
  std::printf("%-24s %8s %12s %10s\n", "policy", "correct", "total regret", "worst");
  const PolicyStat* all[] = {&st_ws, &st_os, &st_k, &st_cm, &st_db, &st_hw};
  for (const PolicyStat* s : all) {
    std::printf("%-24s %5d/%-3d %12ld %9.1f%%\n",
                s->name, s->correct, points, s->regret, s->worst);
  }
  h.trace_en = true;
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Verilated::traceEverOn(true);

  bool sweep_only = false;
  for (int i = 1; i < argc; i++)
    if (std::string(argv[i]) == "--sweep") sweep_only = true;

  Vaccel_top* dut = new Vaccel_top;
  VerilatedVcdC* tfp = new VerilatedVcdC;
  dut->trace(tfp, 99);
  tfp->open("waveform.vcd");

  Harness h(dut, tfp);
  h.reset();

  if (sweep_only) {
    std::printf("=== WS vs OS measurement sweep (N_ARR=%d, STREAM_DEPTH=%d) ===\n",
                N_ARR, STREAM_DEPTH);
    run_sweep(h);
    tfp->close();
    dut->final();
    delete dut; delete tfp;
    if (g_failures == 0) std::printf("\nSWEEP COMPLETE\n");
    else                 std::printf("\n%d CHECK(S) FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
  }

  std::printf("=== Dual-dataflow systolic accelerator ===\n");
  std::printf("N_ARR=%d  STREAM_DEPTH=%d\n\n", N_ARR, STREAM_DEPTH);

  // ---- Directed value coverage on the exact-fit shape ----
  std::printf("-- directed value coverage (4x4x4) --\n");
  {
    std::vector<int8_t> A = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    std::vector<int8_t> Bi = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    shape_test(h, "identity", 4, 4, 4, A, Bi);

    std::vector<int8_t> An(16, -1), Bn(16, -1);
    for (int i = 0; i < 16; i++) An[i] = static_cast<int8_t>(-(i + 1));
    shape_test(h, "negative", 4, 4, 4, An, Bn);

    std::vector<int8_t> Az(16, 0), Bz = {5,-3,7,1, 2,4,-8,6, 9,-2,3,-5, 1,1,1,1};
    shape_test(h, "zeros", 4, 4, 4, Az, Bz);

    std::vector<int8_t> Am = {1,-2,3,-4,-5,6,-7,8,9,-10,11,-12,-13,14,-15,16};
    std::vector<int8_t> Bm = {-1,2,-3,4,5,-6,7,-8,-9,10,-11,12,13,-14,15,-16};
    shape_test(h, "mixed_signs", 4, 4, 4, Am, Bm);

    std::vector<int8_t> Ax = {127,-128,127,-128,-128,127,-128,127,
                              127,127,-128,-128,-128,-128,127,127};
    std::vector<int8_t> Bx = {127,127,-128,-128,-128,127,127,-128,
                              127,-128,127,-128,-128,-128,-128,127};
    shape_test(h, "int8_extremes", 4, 4, 4, Ax, Bx);
  }

  // ---- Tiling / ragged / multi-K shape coverage ----
  std::printf("\n-- tiling and ragged shapes --\n");
  {
    struct Shape { const char* name; int M, N, K; };
    const Shape shapes[] = {
      {"exact_fit",        4,  4,  4},   // one tile, no raggedness
      {"multi_n_tile",     4,  8,  4},   // N tiling only
      {"multi_k_tile",     4,  4,  8},   // K tiling -> exercises C RMW
      {"multi_nk",         4,  8,  8},
      {"ragged_n",         4,  6,  4},   // N tiles [4,2]
      {"ragged_k",         4,  4,  6},   // K tiles [4,2]
      {"ragged_all",       5,  5,  5},   // ragged in every dimension
      {"minimal_1x1x1",    1,  1,  1},   // k_tile=n_tile=m_chunk=1
      {"thin_row",         1, 10, 13},
      {"thin_col",        20,  1,  1},
      {"m_chunking",      17,  4,  4},   // M chunks [16,1]
      {"m_chunk_ragged",  33,  4,  4},   // M chunks [16,16,1]
      {"deep_k",           4,  4, 20},   // 5 K tiles, 4 RMW passes
      {"spec_example",    37, 10, 13},   // from the architecture spec
    };
    for (const auto& s : shapes) {
      std::vector<int8_t> A, B;
      fill_random(A, static_cast<size_t>(s.M) * s.K);
      fill_random(B, static_cast<size_t>(s.K) * s.N);
      h.trace_en = (s.M * s.N * s.K <= 256);  // keep the VCD small
      shape_test(h, s.name, s.M, s.N, s.K, A, B);
    }
    h.trace_en = true;
  }

  // ---- Output-stationary correctness over the same shape space ----
  std::printf("\n-- output-stationary shapes --\n");
  {
    struct Shape { const char* name; int M, N, K; };
    const Shape shapes[] = {
      {"os_exact_fit",     4,  4,  4},
      {"os_multi_m",       8,  4,  4},   // M tiling (OS maps M spatially)
      {"os_multi_n",       4,  8,  4},
      {"os_deep_k",        4,  4, 20},   // one clear, no RMW at all
      {"os_k_chunking",    4,  4, 20},   // K past STREAM_DEPTH
      {"os_ragged_m",      6,  4,  4},
      {"os_ragged_n",      4,  6,  4},
      {"os_ragged_all",    5,  5,  5},
      {"os_minimal",       1,  1,  1},
      {"os_thin_row",      1, 10, 13},
      {"os_thin_col",     20,  1,  1},
      {"os_spec_example", 37, 10, 13},
    };
    for (const auto& s : shapes) {
      std::vector<int8_t> A, B;
      fill_random(A, static_cast<size_t>(s.M) * s.K);
      fill_random(B, static_cast<size_t>(s.K) * s.N);
      h.trace_en = (s.M * s.N * s.K <= 256);
      shape_test(h, s.name, s.M, s.N, s.K, A, B, true, FORCE_OS);
    }
    h.trace_en = true;
  }

  // ---- ADAPTIVE: the hardware must choose, and choose correctly ----
  std::printf("\n-- adaptive policy --\n");
  {
    int before = g_failures;
    struct AdCase { const char* name; int M, N, K; bool expect_os; };
    // Expectations come from the Milestone 5 measurements, not from
    // re-running the same formula the RTL uses -- otherwise this would
    // only prove the hardware agrees with itself.
    const AdCase cases[] = {
      {"small_square",  4,  4,  4, false},  // tie; either is acceptable
      {"deep_k",        4,  4, 32, true },  // OS by 48%
      {"tall_thin_m",  64,  4,  4, false},  // WS by 63%
      {"wide_k",        1, 16, 64, true },  // OS by 38%
      {"big_m_small_k",64, 16,  8, false},  // WS by 5.8%
      {"big_m_deep_k", 64,  4, 32, true },  // OS by 24%
    };
    for (const auto& c : cases) {
      std::vector<int8_t> A, B;
      fill_random(A, static_cast<size_t>(c.M) * c.K);
      fill_random(B, static_cast<size_t>(c.K) * c.N);
      std::vector<int32_t> gold;
      golden(c.M, c.N, c.K, A, B, gold);
      h.trace_en = false;
      PhaseCounts p = h.run_gemm(c.M, c.N, c.K, A, B, ADAPTIVE);
      h.trace_en = true;
      bool chose_os = (h.perf(PERF_MODE) != 0);
      check(h.mem.c == gold, std::string(c.name) + ": ADAPTIVE result wrong");
      // The 4x4x4 case is a genuine tie, so only the decisive shapes
      // are held to a specific choice.
      if (c.M != 4 || c.N != 4 || c.K != 4) {
        check(chose_os == c.expect_os,
              std::string(c.name) + ": ADAPTIVE chose " + (chose_os ? "OS" : "WS") +
                  ", measurements favour " + (c.expect_os ? "OS" : "WS"));
      }
      std::printf("  %-16s %3dx%3dx%3d  chose %s  cyc=%d\n",
                  c.name, c.M, c.N, c.K, chose_os ? "OS" : "WS", p.total());
    }
    std::printf("  %-22s %s\n", "adaptive_choices",
                g_failures == before ? "PASS" : "FAIL");
  }

  // ---- Randomized shape sweep, both dataflows on identical data ----
  std::printf("\n-- randomized shapes (WS and OS) --\n");
  {
    int before = g_failures;
    const int ITERS = 120;
    h.trace_en = false;
    for (int it = 0; it < ITERS; it++) {
      int M = 1 + (rnd() % 24);
      int Nn = 1 + (rnd() % 12);
      int K = 1 + (rnd() % 14);
      std::vector<int8_t> A, B;
      fill_random(A, static_cast<size_t>(M) * K);
      fill_random(B, static_cast<size_t>(K) * Nn);
      // Both dataflows must produce identical results from identical
      // inputs; each is independently checked against the golden model.
      shape_test(h, "rand_ws", M, Nn, K, A, B, false, FORCE_WS);
      shape_test(h, "rand_os", M, Nn, K, A, B, false, FORCE_OS);
      if (g_failures != before) {
        std::printf("  first failing shape: %dx%dx%d\n", M, Nn, K);
        break;
      }
    }
    h.trace_en = true;
    std::printf("  %-22s %s  (%d random shapes x 2 dataflows)\n", "randomized_shapes",
                g_failures == before ? "PASS" : "FAIL", ITERS);
  }

  // ---- Reset behaviour ----
  std::printf("\n-- control and error behaviour --\n");
  {
    int before = g_failures;
    std::vector<int8_t> A, B;
    fill_random(A, 8 * 8);
    fill_random(B, 8 * 8);
    h.mem.a = A; h.mem.b = B;
    h.mem.c.assign(64, C_POISON);
    dut->cfg_m = 8; dut->cfg_n = 8; dut->cfg_k = 8; dut->cfg_policy = 0;
    dut->start = 1; h.tick(); dut->start = 0;
    for (int i = 0; i < 25; i++) h.tick();
    check(!dut->done, "reset test: done fired unexpectedly early");
    h.reset();
    check(!dut->busy, "reset test: busy still asserted after reset");
    check(!dut->done, "reset test: done still asserted after reset");
    shape_test(h, "reset_recovery", 8, 8, 8, A, B, /*verbose=*/false);
    std::printf("  %-22s %s\n", "reset_mid_operation", g_failures == before ? "PASS" : "FAIL");
  }

  // Back-to-back with no intervening reset.
  {
    int before = g_failures;
    std::vector<int8_t> A1, B1, A2, B2;
    fill_random(A1, 6 * 5); fill_random(B1, 5 * 7);
    fill_random(A2, 9 * 3); fill_random(B2, 3 * 4);
    shape_test(h, "b2b_first",  6, 7, 5, A1, B1, /*verbose=*/false);
    shape_test(h, "b2b_second", 9, 4, 3, A2, B2, /*verbose=*/false);
    std::printf("  %-22s %s\n", "back_to_back_runs", g_failures == before ? "PASS" : "FAIL");
  }

  // Policy encodings: only FORCE_WS is implemented.
  {
    int before = g_failures;
    struct PC { int policy; const char* name; bool err; };
    // All three real policies are implemented; RESERVED must still be
    // rejected rather than aliased onto one of them.
    const PC cases[] = {{0,"FORCE_WS",false},{1,"FORCE_OS",false},
                        {2,"ADAPTIVE",false},{3,"RESERVED",true}};
    for (const auto& pc : cases) {
      h.reset();
      dut->cfg_m = 4; dut->cfg_n = 4; dut->cfg_k = 4;
      dut->cfg_policy = pc.policy;
      dut->start = 1; h.tick(); dut->start = 0;
      std::string nm = std::string("policy ") + pc.name;
      if (pc.err) {
        check(dut->error, nm + ": expected error");
        check(!dut->busy, nm + ": must not assert busy");
        check(!dut->done, nm + ": must not assert done");
      } else {
        check(!dut->error, nm + ": unexpected error");
        check(dut->busy,   nm + ": expected busy");
      }
      h.tick();
    }
    std::printf("  %-22s %s\n", "policy_encoding_matrix", g_failures == before ? "PASS" : "FAIL");
  }

  // Illegal shapes must be rejected, not silently mis-computed.
  {
    int before = g_failures;
    const int bad[][3] = {{0,4,4},{4,0,4},{4,4,0},{256,4,4},{4,256,4},{4,4,257}};
    for (const auto& s : bad) {
      h.reset();
      dut->cfg_m = s[0]; dut->cfg_n = s[1]; dut->cfg_k = s[2];
      dut->cfg_policy = 0;
      dut->start = 1; h.tick(); dut->start = 0;
      std::string nm = "shape " + std::to_string(s[0]) + "x" + std::to_string(s[1]) +
                       "x" + std::to_string(s[2]);
      check(dut->error, nm + ": expected error");
      check(!dut->busy, nm + ": must not assert busy");
      h.tick();
    }
    std::printf("  %-22s %s\n", "illegal_shape_errors", g_failures == before ? "PASS" : "FAIL");
  }

  tfp->close();
  dut->final();
  delete dut;
  delete tfp;

  if (g_failures == 0) std::printf("\nALL TESTS PASSED\n");
  else                 std::printf("\n%d CHECK(S) FAILED\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}
