// Phase 2 — the drain-time baseline: the number the yield has to beat
// (ROADMAP: "how long you wait when the only option is to let the current
// work drain").
//
// A persistent background kernel chews through a long queue of tiles. At a
// random point an urgent single-tile kernel is launched from the host on a
// max-priority stream. No preemption exists yet, so the urgent block runs
// only when a residency slot frees up. The measurement is how long it waits.
// Result on the A10: 957 us p50 at saturation, r=0.99 with remaining queue
// — the figure yield.cu's e2e is compared against.
//
// Run it (scripts/phase2.sh drives all three shapes):
//   ./bin/urgent_baseline --mode sat --trials 2000 --csv results/raw/x.csv
//
// The load-bearing subtlety: "the GPU is busy" is not one condition.
//   --mode sat    grid = SMs x max resident blocks/SM  -> every residency
//                 slot occupied; urgent must wait for drain. TRUE baseline.
//   --mode sat-1  one slot left open on one SM         -> urgent should
//                 start ~immediately despite 100% "utilization".
//   --mode 1persm grid = 72 (Phase 1 shape)            -> most slots free.
// Phase 1's 72-block launch does NOT saturate residency; a baseline measured
// that way would be silently wrong. Stream priority cannot help in sat mode:
// priorities reorder pending blocks, they do not evict resident ones.
//
// All cross-kernel timing uses %globaltimer (ns wall clock, coherent across
// SMs and kernels, independent of the SM clock lock). clock64() would be
// wrong here: different SMs, different counters.
//
// Metrics per trial:
//   e2e_us   host arrival -> urgent completion observed on host (the number
//            a caller experiences, includes launch + sync overhead)
//   gap_us   urgent device start - bg last-block exit (device timeline).
//            sat: expect within one tile of 0. sat-1/1persm: large negative
//            (urgent starts long before bg drains).
//   exec_us  urgent kernel device duration
//   drain_us bg first-block start -> last-block exit (device timeline)
// Remaining queue length at arrival is snapshotted so e2e can be checked
// against remaining-work-at-arrival, not just against averages.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#include "tile.cuh"

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err_ = (call);                                                \
    if (err_ != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err_));                                      \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

__global__ void __launch_bounds__(THREADS) persistent_bg(
    const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int K, int N, int tilesN, int totalTiles,
    int numTasks, unsigned int* nextTask, int* exited, int gridBlocks,
    long long* bgStartGT, long long* bgEndGT) {
  if (blockIdx.x == 0 && threadIdx.x == 0) *bgStartGT = globaltimer_ns();
  __shared__ unsigned s_task;

  while (true) {
    if (threadIdx.x == 0) s_task = atomicAdd(nextTask, 1u);
    __syncthreads();
    const unsigned task = s_task;
    __syncthreads();
    if (task >= (unsigned)numTasks) break;

    // Tasks cycle over the tile grid; repeated writes of a tile are
    // idempotent (same A, B), so verification stays valid at any task count.
    const int tile = (int)(task % (unsigned)totalTiles);
    const int r = (tile / tilesN) * TM;
    const int c = (tile % tilesN) * TN;
    tile_gemm(A, B, C, K, N, r, c, N, r, c);
    __syncthreads();
  }

  if (threadIdx.x == 0 && atomicAdd(exited, 1) == gridBlocks - 1)
    *bgEndGT = globaltimer_ns();
}

// A3 reserved arm: the urgent job is U blocks, block b computing tile b —
// distinct real work per block, written compactly per block. uEndGT is an
// atomicMax so it holds the LAST block's completion (job-level e2e).
//
// The launch-bounds min-blocks pin is load-bearing (fault 13): without it
// this kernel compiled to 79 registers, which does not fit the 16,384
// registers a 3-way-occupied SM has spare, so the urgent job waited for
// full drain DESPITE reserved slots. Reservation is denominated in
// resource envelopes, not slot counts; the pin caps this kernel at 64
// registers so it fits the envelope its own reservation frees.
__global__ void __launch_bounds__(THREADS, 4) urgent_tile(
    const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ Cu, int K, int N, int tilesN, long long* uStartGT,
    unsigned long long* uEndGT) {
  if (blockIdx.x == 0 && threadIdx.x == 0) *uStartGT = globaltimer_ns();
  const int t = (int)blockIdx.x;
  tile_gemm(A, B, Cu + (size_t)t * TM * TN, K, N, (t / tilesN) * TM,
            (t % tilesN) * TN, TN, 0, 0);
  __syncthreads();
  if (threadIdx.x == 0)
    atomicMax(uEndGT, (unsigned long long)globaltimer_ns());
}

// ---------------------------------------------------------------------------

static double parse_arg(int argc, char** argv, const char* name, double dflt) {
  for (int i = 1; i + 1 < argc; ++i)
    if (strcmp(argv[i], name) == 0) return atof(argv[i + 1]);
  return dflt;
}

static const char* parse_str(int argc, char** argv, const char* name,
                             const char* dflt) {
  for (int i = 1; i + 1 < argc; ++i)
    if (strcmp(argv[i], name) == 0) return argv[i + 1];
  return dflt;
}

static double now_us() {
  using namespace std::chrono;
  return duration<double, std::micro>(steady_clock::now().time_since_epoch())
      .count();
}

static void spin_until_us(double deadline) {
  while (now_us() < deadline) { /* spin; ms-scale delays need us precision */ }
}

static double percentile(std::vector<double> v, double q) {
  std::sort(v.begin(), v.end());
  return v[std::min((size_t)(q * v.size()), v.size() - 1)];
}

int main(int argc, char** argv) {
  const int K = (int)parse_arg(argc, argv, "--k", 64);
  const int tilesM = (int)parse_arg(argc, argv, "--tiles-m", 48);
  const int tilesN = (int)parse_arg(argc, argv, "--tiles-n", 48);
  const int bgTasks = (int)parse_arg(argc, argv, "--bg-tasks", 20000);
  const int trials = (int)parse_arg(argc, argv, "--trials", 500);
  const std::string mode = parse_str(argc, argv, "--mode", "sat");
  int blocks = (int)parse_arg(argc, argv, "--blocks", 0);
  const int reserve = (int)parse_arg(argc, argv, "--reserve", -1);
  const int urgentBlocks = (int)parse_arg(argc, argv, "--urgent-blocks", 1);
  const char* csvPath = parse_str(argc, argv, "--csv", "");

  if (K % KB != 0) {
    fprintf(stderr, "--k must be a multiple of %d\n", KB);
    return 1;
  }

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  const int sms = prop.multiProcessorCount;

  int maxPerSM = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxPerSM, persistent_bg, THREADS, 0));

  if (blocks <= 0) {
    if (reserve >= 0) blocks = sms * maxPerSM - reserve;
    else if (mode == "sat") blocks = sms * maxPerSM;
    else if (mode == "sat-1") blocks = sms * maxPerSM - 1;
    else if (mode == "1persm") blocks = sms;
    else { fprintf(stderr, "unknown --mode %s\n", mode.c_str()); return 1; }
  }

  const int M = tilesM * TM;
  const int N = tilesN * TN;
  const int totalTiles = tilesM * tilesN;

  printf("GPU: %s | %d SMs | %d resident blocks/SM max -> %d slots\n",
         prop.name, sms, maxPerSM, sms * maxPerSM);
  printf("mode=%s: %d blocks | reserve=%d urgent-blocks=%d | K=%d | "
         "bg-tasks=%d | trials=%d\n",
         mode.c_str(), blocks, reserve, urgentBlocks, K, bgTasks, trials);

  srand(21);
  std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
  for (auto& v : hA) v = (float)(rand() / (double)RAND_MAX * 2.0 - 1.0);
  for (auto& v : hB) v = (float)(rand() / (double)RAND_MAX * 2.0 - 1.0);

  float *dA, *dB, *dC, *dCu;
  unsigned int* dNext;
  int* dExited;
  long long *dBgStart, *dBgEnd, *dUStart, *dUEnd;
  CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, (size_t)M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dCu,
                        (size_t)urgentBlocks * TM * TN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dNext, sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc(&dExited, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dBgStart, sizeof(long long)));
  CUDA_CHECK(cudaMalloc(&dBgEnd, sizeof(long long)));
  CUDA_CHECK(cudaMalloc(&dUStart, sizeof(long long)));
  CUDA_CHECK(cudaMalloc(&dUEnd, sizeof(long long)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  unsigned int* hConsumed;  // pinned, for the mid-flight queue snapshot
  CUDA_CHECK(cudaMallocHost(&hConsumed, sizeof(unsigned int)));

  int loPri, hiPri;
  CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&loPri, &hiPri));
  cudaStream_t bgStream, uStream;
  CUDA_CHECK(cudaStreamCreateWithPriority(&bgStream, cudaStreamNonBlocking,
                                          loPri));
  CUDA_CHECK(cudaStreamCreateWithPriority(&uStream, cudaStreamNonBlocking,
                                          hiPri));
  printf("stream priorities: bg=%d urgent=%d (lower value = higher)\n", loPri,
         hiPri);

  auto reset_run = [&](cudaStream_t s) {
    CUDA_CHECK(cudaMemsetAsync(dNext, 0, sizeof(unsigned int), s));
    CUDA_CHECK(cudaMemsetAsync(dExited, 0, sizeof(int), s));
    CUDA_CHECK(cudaMemsetAsync(dBgStart, 0, sizeof(long long), s));
    CUDA_CHECK(cudaMemsetAsync(dBgEnd, 0, sizeof(long long), s));
  };
  auto launch_bg = [&]() {
    persistent_bg<<<blocks, THREADS, 0, bgStream>>>(
        dA, dB, dC, K, N, tilesN, totalTiles, bgTasks, dNext, dExited, blocks,
        dBgStart, dBgEnd);
  };
  auto read_ll = [&](long long* dptr) {
    long long v;
    CUDA_CHECK(cudaMemcpy(&v, dptr, sizeof(v), cudaMemcpyDeviceToHost));
    return v;
  };

  // --- Correctness gate (once, before any timing): both kernels vs float64.
  {
    printf("verifying bg + urgent kernels against float64 host reference...\n");
    CUDA_CHECK(cudaMemset(dC, 0xFF, (size_t)M * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(dCu, 0xFF,
                        (size_t)urgentBlocks * TM * TN * sizeof(float)));
    reset_run(bgStream);
    // Null-stream fills don't order against non-blocking streams, and a
    // saturated grid blocks the fill kernel outright; sync before launch.
    CUDA_CHECK(cudaDeviceSynchronize());
    launch_bg();
    CUDA_CHECK(cudaMemsetAsync(dUEnd, 0, sizeof(long long), uStream));
    urgent_tile<<<urgentBlocks, THREADS, 0, uStream>>>(
        dA, dB, dCu, K, N, tilesN, dUStart, (unsigned long long*)dUEnd);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    std::vector<float> hC((size_t)M * N),
        hCu((size_t)urgentBlocks * TM * TN);
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hCu.data(), dCu, hCu.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    // bg covers the whole tile grid only if bgTasks >= totalTiles. The
    // urgent job's U tiles are checked against the same per-element ref.
    const int coveredTiles = std::min(bgTasks, totalTiles);
    int bad = 0, nans = 0;
    double maxRel = 0.0;
    const double rtol = 1e-4, atol = 1e-4;
    for (int tile = 0; tile < coveredTiles; ++tile) {
      const int r0 = (tile / tilesN) * TM, c0 = (tile % tilesN) * TN;
      for (int i = 0; i < TM; ++i) {
        const int row = r0 + i;
        for (int j = 0; j < TN; ++j) {
          const int col = c0 + j;
          double ref = 0.0;
          for (int k = 0; k < K; ++k)
            ref += (double)hA[(size_t)row * K + k] * hB[(size_t)k * N + col];
          auto check = [&](float got) {
            if (std::isnan(got)) { ++nans; return; }
            const double err = std::fabs(got - ref);
            maxRel = std::max(maxRel, err / std::max(1.0, std::fabs(ref)));
            if (err > atol + rtol * std::fabs(ref)) ++bad;
          };
          check(hC[(size_t)row * N + col]);
          if (tile < urgentBlocks)
            check(hCu[(size_t)tile * TM * TN + (size_t)i * TN + j]);
        }
      }
    }
    if (nans || bad) {
      fprintf(stderr, "FAIL: %d NaN, %d beyond tolerance, max rel %.3e\n",
              nans, bad, maxRel);
      return 1;
    }
    printf("PASS: %d tiles + %d urgent tiles, max rel err %.3e\n",
           coveredTiles, urgentBlocks, maxRel);
  }

  // --- Drain estimate (bg alone, device timeline), used to place arrivals.
  reset_run(bgStream);
  launch_bg();
  CUDA_CHECK(cudaStreamSynchronize(bgStream));
  const double drainEstUs = (read_ll(dBgEnd) - read_ll(dBgStart)) / 1000.0;
  printf("bg drain, no urgent: %.0f us for %d tasks on %d blocks\n",
         drainEstUs, bgTasks, blocks);

  // --- Trials. Arrival at U(0.2, 0.8) of estimated drain, fixed seed.
  std::mt19937 rng(21);
  std::uniform_real_distribution<double> frac(0.2, 0.8);

  struct Row {
    double delayUs, e2eUs, execUs, gapUs, drainUs;
    unsigned consumed;
  };
  std::vector<Row> rows;
  rows.reserve(trials);

  for (int t = 0; t < trials; ++t) {
    reset_run(bgStream);
    const double delayUs = frac(rng) * drainEstUs;
    const double t0 = now_us();
    launch_bg();
    spin_until_us(t0 + delayUs);

    const double tArrive = now_us();
    CUDA_CHECK(cudaMemcpyAsync(hConsumed, dNext, sizeof(unsigned int),
                               cudaMemcpyDeviceToHost, uStream));
    CUDA_CHECK(cudaMemsetAsync(dUEnd, 0, sizeof(long long), uStream));
    urgent_tile<<<urgentBlocks, THREADS, 0, uStream>>>(
        dA, dB, dCu, K, N, tilesN, dUStart, (unsigned long long*)dUEnd);
    CUDA_CHECK(cudaStreamSynchronize(uStream));
    const double tDone = now_us();
    CUDA_CHECK(cudaStreamSynchronize(bgStream));
    CUDA_CHECK(cudaGetLastError());

    const long long uS = read_ll(dUStart), uE = read_ll(dUEnd);
    const long long bS = read_ll(dBgStart), bE = read_ll(dBgEnd);
    rows.push_back({delayUs, tDone - tArrive, (uE - uS) / 1000.0,
                    (uS - bE) / 1000.0, (bE - bS) / 1000.0,
                    std::min(*hConsumed, (unsigned)bgTasks)});
  }

  std::vector<double> e2e, gap, exec, drain;
  for (auto& r : rows) {
    e2e.push_back(r.e2eUs);
    gap.push_back(r.gapUs);
    exec.push_back(r.execUs);
    drain.push_back(r.drainUs);
  }
  printf("\n--- %s: %d trials ---\n", mode.c_str(), trials);
  printf("urgent e2e (us):  p50 %.1f | p90 %.1f | p99 %.1f | max %.1f\n",
         percentile(e2e, 0.50), percentile(e2e, 0.90), percentile(e2e, 0.99),
         *std::max_element(e2e.begin(), e2e.end()));
  printf("gap uStart-bgEnd (us): p50 %.1f | p10 %.1f | p90 %.1f\n",
         percentile(gap, 0.50), percentile(gap, 0.10), percentile(gap, 0.90));
  printf("urgent exec (us): p50 %.2f | p99 %.2f\n", percentile(exec, 0.50),
         percentile(exec, 0.99));
  printf("bg drain (us):    p50 %.0f\n", percentile(drain, 0.50));

  if (csvPath[0]) {
    FILE* csv = fopen(csvPath, "w");
    if (csv) {
      fprintf(csv,
              "trial,delay_us,consumed_at_arrival,remaining_at_arrival,"
              "e2e_us,exec_us,gap_us,drain_us\n");
      for (int t = 0; t < trials; ++t)
        fprintf(csv, "%d,%.1f,%u,%d,%.1f,%.2f,%.1f,%.1f\n", t,
                rows[t].delayUs, rows[t].consumed,
                bgTasks - (int)rows[t].consumed, rows[t].e2eUs, rows[t].execUs,
                rows[t].gapUs, rows[t].drainUs);
      fclose(csv);
      printf("raw trial records: %s\n", csvPath);
    } else {
      fprintf(stderr, "warning: could not open %s\n", csvPath);
    }
  }
  return 0;
}
