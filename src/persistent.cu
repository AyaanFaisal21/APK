// Phase 1 — the minimal persistent kernel, and the tile calibrator every
// later phase leans on.
//
// Premise: a persistent kernel is a thread pool — blocks start once and
// pull tile tasks off a global fetch-add queue until it drains. Each task
// is one TM x TN tile of a large SGEMM (the tile itself lives in tile.cuh,
// shared with the other binaries), with K as the duration knob. This file
// established the K -> duration map (K in {32,64,128} spans 6.7-17.8 us at
// 1200 MHz) that phase2/3/4 configs quote.
//
// Run it (scripts/calibrate.sh drives the sweep):
//   ./bin/persistent --sm-clock-mhz 1200                 # headline config
//   ./bin/persistent --blocks 288 --k 64 --verify 0      # co-residency walls
//
// Termination is by queue exhaustion, so correctness does not depend on all
// blocks being co-resident: even if blocks serialized onto one SM the run
// would still complete. No inter-block waiting exists in Phase 1 — deadlock
// becomes possible only when dependencies enter the queue (later phases).
//
// Timing is on-device clock64() per tile — same-SM start/end, valid for
// durations, but cycles convert to us at the LOCKED clock, and a clock
// lock is only evidence under load (the 150 W cap throttles straight
// through a 1695 MHz lock; NOTEBOOK 2026-08-04). Pass the verified value
// via --sm-clock-mhz. Host wall time is reported separately via cudaEvents.
//
// Guard against the Cornfield stale-buffer failure mode: C and the tile log
// are filled with 0xFF bytes (NaN / -1 patterns) before every pass, so a
// tile that never executes fails verification loudly instead of passing
// allclose off recycled memory.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
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

__global__ void __launch_bounds__(THREADS) persistent_gemm(
    const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int K, int N, int tilesN, int numTasks,
    unsigned int* nextTask, TileRecord* log) {
  __shared__ unsigned s_task;

  while (true) {
    if (threadIdx.x == 0) s_task = atomicAdd(nextTask, 1u);
    __syncthreads();
    const unsigned task = s_task;
    __syncthreads();  // everyone has read s_task before it can be rewritten
    if (task >= (unsigned)numTasks) break;

    const long long t0 = clock64();

    const int rowBase = (task / tilesN) * TM;
    const int colBase = (task % tilesN) * TN;
    tile_gemm(A, B, C, K, N, rowBase, colBase, N, rowBase, colBase);

    __syncthreads();  // whole tile committed before the clock closes
    const long long t1 = clock64();
    if (threadIdx.x == 0) {
      log[task].start = t0;
      log[task].end = t1;
      log[task].smid = (int)smid_of();
      log[task].block = (int)blockIdx.x;
    }
  }
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

int main(int argc, char** argv) {
  const int K = (int)parse_arg(argc, argv, "--k", 128);
  const int tilesM = (int)parse_arg(argc, argv, "--tiles-m", 48);
  const int tilesN = (int)parse_arg(argc, argv, "--tiles-n", 48);
  const int passes = (int)parse_arg(argc, argv, "--passes", 3);
  const int verify = (int)parse_arg(argc, argv, "--verify", 1);
  int blocks = (int)parse_arg(argc, argv, "--blocks", 0);
  double smClockMHz = parse_arg(argc, argv, "--sm-clock-mhz", 0.0);
  const char* csvPath =
      parse_str(argc, argv, "--csv", "results/raw/phase1_tiles.csv");

  if (K % KB != 0) {
    fprintf(stderr, "--k must be a multiple of %d\n", KB);
    return 1;
  }

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  if (blocks <= 0) blocks = prop.multiProcessorCount;
  if (smClockMHz <= 0.0) smClockMHz = prop.clockRate / 1000.0;  // max boost

  const int M = tilesM * TM;
  const int N = tilesN * TN;
  const int numTasks = tilesM * tilesN;
  const double flopsPerTile = 2.0 * TM * TN * K;

  printf("GPU: %s | %d SMs | cc %d.%d\n", prop.name,
         prop.multiProcessorCount, prop.major, prop.minor);
  printf("config: M=%d N=%d K=%d | tile %dx%d | tasks=%d | blocks=%d x %d "
         "threads | passes=%d\n",
         M, N, K, TM, TN, numTasks, blocks, THREADS, passes);
  printf("clock for us conversion: %.0f MHz "
         "(pass --sm-clock-mhz to match locked clocks)\n",
         smClockMHz);

  // Deterministic host data in [-1, 1).
  srand(21);
  std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
  for (auto& v : hA) v = (float)(rand() / (double)RAND_MAX * 2.0 - 1.0);
  for (auto& v : hB) v = (float)(rand() / (double)RAND_MAX * 2.0 - 1.0);

  float *dA, *dB, *dC;
  unsigned int* dNext;
  TileRecord* dLog;
  CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, (size_t)M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dNext, sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc(&dLog, (size_t)numTasks * sizeof(TileRecord)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  cudaEvent_t evStart, evStop;
  CUDA_CHECK(cudaEventCreate(&evStart));
  CUDA_CHECK(cudaEventCreate(&evStop));

  float lastPassMs = 0.0f;
  for (int p = 0; p < passes; ++p) {
    // 0xFF fill: C becomes NaN, log.block becomes -1. A tile that never
    // runs cannot pass verification off stale memory.
    CUDA_CHECK(cudaMemset(dC, 0xFF, (size_t)M * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(dLog, 0xFF, (size_t)numTasks * sizeof(TileRecord)));
    CUDA_CHECK(cudaMemset(dNext, 0, sizeof(unsigned int)));

    CUDA_CHECK(cudaEventRecord(evStart));
    persistent_gemm<<<blocks, THREADS>>>(dA, dB, dC, K, N, tilesN, numTasks,
                                         dNext, dLog);
    CUDA_CHECK(cudaEventRecord(evStop));
    CUDA_CHECK(cudaEventSynchronize(evStop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventElapsedTime(&lastPassMs, evStart, evStop));
    printf("pass %d: %.3f ms wall | %.1f GFLOP/s | %.0f tasks/s%s\n", p,
           lastPassMs, flopsPerTile * numTasks / (lastPassMs * 1e6),
           numTasks / (lastPassMs * 1e-3), p + 1 < passes ? " (warmup)" : "");
  }

  std::vector<TileRecord> hLog(numTasks);
  CUDA_CHECK(cudaMemcpy(hLog.data(), dLog, numTasks * sizeof(TileRecord),
                        cudaMemcpyDeviceToHost));

  // Completion check: every task must have been logged exactly by a worker.
  int unlogged = 0;
  for (int t = 0; t < numTasks; ++t)
    if (hLog[t].block < 0) ++unlogged;
  if (unlogged) {
    fprintf(stderr, "FAIL: %d of %d tasks never logged a record\n", unlogged,
            numTasks);
    return 1;
  }

  // Duration distribution (final pass only; earlier passes are warmup).
  std::vector<double> durUs(numTasks);
  for (int t = 0; t < numTasks; ++t)
    durUs[t] = (hLog[t].end - hLog[t].start) / smClockMHz;  // cycles/MHz = us
  std::vector<double> sorted = durUs;
  std::sort(sorted.begin(), sorted.end());
  auto pct = [&](double q) {
    return sorted[std::min((size_t)(q * sorted.size()), sorted.size() - 1)];
  };
  double mean = 0;
  for (double d : sorted) mean += d;
  mean /= sorted.size();
  printf("tile duration (us, final pass, n=%d): min %.2f | p50 %.2f | "
         "p90 %.2f | p99 %.2f | max %.2f | mean %.2f\n",
         numTasks, sorted.front(), pct(0.50), pct(0.90), pct(0.99),
         sorted.back(), mean);
  const bool inBand = pct(0.50) >= 2.0 && pct(0.50) <= 25.0;
  printf("2-25 us band check (p50): %s\n", inBand ? "IN BAND" : "OUT OF BAND");

  // Tasks-per-block spread — sanity check that the queue actually balanced.
  std::vector<int> perBlock(blocks, 0);
  for (int t = 0; t < numTasks; ++t) ++perBlock[hLog[t].block];
  printf("tasks/block: min %d | max %d (ideal %.1f)\n",
         *std::min_element(perBlock.begin(), perBlock.end()),
         *std::max_element(perBlock.begin(), perBlock.end()),
         (double)numTasks / blocks);

  // Verification against a float64 host reference (never another GPU run).
  if (verify) {
    std::vector<float> hC((size_t)M * N);
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    printf("verifying against float64 host reference...\n");
    std::vector<double> ref((size_t)M * N, 0.0);
    for (int i = 0; i < M; ++i)
      for (int k = 0; k < K; ++k) {
        const double a = hA[(size_t)i * K + k];
        const size_t crow = (size_t)i * N, brow = (size_t)k * N;
        for (int j = 0; j < N; ++j) ref[crow + j] += a * hB[brow + j];
      }
    int bad = 0, nans = 0;
    double maxRel = 0.0;
    const double rtol = 1e-4, atol = 1e-4;
    for (size_t i = 0; i < ref.size(); ++i) {
      if (std::isnan(hC[i])) { ++nans; continue; }
      const double err = std::fabs(hC[i] - ref[i]);
      const double rel = err / std::max(1.0, std::fabs(ref[i]));
      maxRel = std::max(maxRel, rel);
      if (err > atol + rtol * std::fabs(ref[i])) ++bad;
    }
    if (nans || bad) {
      fprintf(stderr, "FAIL: %d NaN (never-written) elements, %d beyond "
              "tolerance, max rel err %.3e\n", nans, bad, maxRel);
      return 1;
    }
    printf("PASS: all %zu elements within rtol=%g atol=%g "
           "(max rel err %.3e)\n", ref.size(), rtol, atol, maxRel);
  } else {
    printf("verification SKIPPED (--verify 0)\n");
  }

  // Raw per-tile records to disk before anything else can go wrong.
  FILE* csv = fopen(csvPath, "w");
  if (csv) {
    fprintf(csv, "task,block,smid,start_cyc,end_cyc,dur_cyc,dur_us\n");
    for (int t = 0; t < numTasks; ++t)
      fprintf(csv, "%d,%d,%d,%lld,%lld,%lld,%.3f\n", t, hLog[t].block,
              hLog[t].smid, hLog[t].start, hLog[t].end,
              hLog[t].end - hLog[t].start, durUs[t]);
    fclose(csv);
    printf("raw tile records: %s\n", csvPath);
  } else {
    fprintf(stderr, "warning: could not open %s for writing\n", csvPath);
  }

  return 0;
}
