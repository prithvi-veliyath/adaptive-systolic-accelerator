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
  S_COMPUTE = 4, S_WB = 5, S_NEXT = 6, S_DONE = 7
};

// C is poisoned before every run. The design performs no pre-zeroing
// pass, so the first K tile must overwrite this value outright; if any
// output still reads as poison, a C write was missed.
static const int32_t C_POISON = 0x5A5A5A5A;

struct PhaseCounts {
  int bfetch = 0, wload = 0, afetch = 0, compute = 0, wb = 0, next = 0, done = 0, other = 0;
  int total() const { return bfetch + wload + afetch + compute + wb + next + done + other; }
  // Cycles the systolic array is actually computing, vs. everything else.
  int useful() const { return compute; }
};

// Analytical model of the WS schedule. Kept deliberately independent of
// the RTL so that a disagreement is informative rather than circular.
static PhaseCounts model_cycles(int M, int Nn, int K) {
  PhaseCounts p;
  for (int n0 = 0; n0 < Nn; n0 += N_ARR) {
    int n_tile = std::min(N_ARR, Nn - n0);
    for (int k0 = 0; k0 < K; k0 += N_ARR) {
      int k_tile = std::min(N_ARR, K - k0);
      bool k_first = (k0 == 0);
      p.bfetch += k_tile * n_tile;
      p.wload  += k_tile;
      for (int m0 = 0; m0 < M; m0 += STREAM_DEPTH) {
        int m_chunk = std::min(STREAM_DEPTH, M - m0);
        p.afetch  += m_chunk * k_tile;
        p.compute += m_chunk + n_tile + N_ARR - 1;
        p.wb      += m_chunk * n_tile * (k_first ? 1 : 2);
        p.next    += 1;
      }
    }
  }
  p.done = 1;
  return p;
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

  void account(PhaseCounts& p) {
    switch (cur_state()) {
      case S_BFETCH:  p.bfetch++;  break;
      case S_WLOAD:   p.wload++;   break;
      case S_AFETCH:  p.afetch++;  break;
      case S_COMPUTE: p.compute++; break;
      case S_WB:      p.wb++;      break;
      case S_NEXT:    p.next++;    break;
      case S_DONE:    p.done++;    break;
      default:        p.other++;   break;
    }
  }

  PhaseCounts run_gemm(int M, int Nn, int K,
                       const std::vector<int8_t>& A, const std::vector<int8_t>& B,
                       int max_cycles = 2000000) {
    mem.a = A;
    mem.b = B;
    mem.c.assign(static_cast<size_t>(M) * Nn, C_POISON);
    c_writes = 0;

    dut->cfg_m = M; dut->cfg_n = Nn; dut->cfg_k = K;
    dut->cfg_policy = 0;  // FORCE_WS
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
static void shape_test(Harness& h, const std::string& name, int M, int Nn, int K,
                       const std::vector<int8_t>& A, const std::vector<int8_t>& B,
                       bool verbose = true) {
  int before = g_failures;
  PhaseCounts got = h.run_gemm(M, Nn, K, A, B);
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

  // Each output is written once per K tile: once directly, then once per
  // read-modify-write pass.
  int k_tiles = (K + N_ARR - 1) / N_ARR;
  int expect_writes = M * Nn * k_tiles;
  check(h.c_writes == expect_writes,
        name + ": C writes = " + std::to_string(h.c_writes) + ", expected " +
            std::to_string(expect_writes));

  PhaseCounts exp = model_cycles(M, Nn, K);
  check(got.bfetch  == exp.bfetch,  name + ": BFETCH "  + std::to_string(got.bfetch)  + " vs model " + std::to_string(exp.bfetch));
  check(got.wload   == exp.wload,   name + ": WLOAD "   + std::to_string(got.wload)   + " vs model " + std::to_string(exp.wload));
  check(got.afetch  == exp.afetch,  name + ": AFETCH "  + std::to_string(got.afetch)  + " vs model " + std::to_string(exp.afetch));
  check(got.compute == exp.compute, name + ": COMPUTE " + std::to_string(got.compute) + " vs model " + std::to_string(exp.compute));
  check(got.wb      == exp.wb,      name + ": WB "      + std::to_string(got.wb)      + " vs model " + std::to_string(exp.wb));
  check(got.next    == exp.next,    name + ": NEXT "    + std::to_string(got.next)    + " vs model " + std::to_string(exp.next));
  check(got.other   == 0,           name + ": cycles outside defined phases");

  if (verbose) {
    double util = 100.0 * got.useful() / got.total();
    std::printf("  %-22s %3dx%3dx%3d  %s  cycles=%5d  [bf %d | wl %d | af %d | cp %d | wb %d | nx %d]"
                "  array-active=%.1f%%\n",
                name.c_str(), M, Nn, K, g_failures == before ? "PASS" : "FAIL",
                got.total(), got.bfetch, got.wload, got.afetch, got.compute,
                got.wb, got.next, util);
  }
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

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Verilated::traceEverOn(true);

  Vaccel_top* dut = new Vaccel_top;
  VerilatedVcdC* tfp = new VerilatedVcdC;
  dut->trace(tfp, 99);
  tfp->open("waveform.vcd");

  Harness h(dut, tfp);
  h.reset();

  std::printf("=== Milestone 2: tiled weight-stationary GEMM ===\n");
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

  // ---- Randomized shape sweep ----
  std::printf("\n-- randomized shapes --\n");
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
      shape_test(h, "rand", M, Nn, K, A, B, /*verbose=*/false);
      if (g_failures != before) {
        std::printf("  first failing shape: %dx%dx%d\n", M, Nn, K);
        break;
      }
    }
    h.trace_en = true;
    std::printf("  %-22s %s  (%d random shapes)\n", "randomized_shapes",
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
    const PC cases[] = {{0,"FORCE_WS",false},{1,"FORCE_OS",true},
                        {2,"ADAPTIVE",true},{3,"RESERVED",true}};
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
