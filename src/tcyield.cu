// The generalization test: cooperative handoff inside a CUTLASS-class
// tensor-core pipeline. One task = one 128x128 output tile of
// C(fp32) = A(fp16) x B(fp16), computed with wmma 16x16x16 fragments
// (8 warps, 32x64 per warp) through a THREE-stage cp.async pipeline in
// ~56 KB of dynamic shared memory per block. Occupancy is expected to
// bind at 1 block/SM (smem and registers both) -- the hostile regime
// the SGEMM instruments never reached.
//
// This is a CUTLASS-CLASS kernel, not the CUTLASS library: the
// hypothesis is about the resource and pipeline regime, and
// instrumenting the library's mainloop would modify it anyway.
//
// The scheduling instrument is the proven midyield skeleton: fetch-add
// queue, dev-set generation events, barrier-protected snapshot
// checkpoints (fault-8 discipline), yield-site logging, drain / naive /
// poison disciplines, oracle counters through device globals (fault-5 /
// fault-18 discipline: measurement variants must not change codegen).
//
// Urgent-tile verification is device-side against a host-uploaded
// float64-derived reference (10k x 64 KB captures would not fit);
// one captured tile per run is additionally verified on the host in
// float64. Background C is host-verified in float64 as always.
//
// Run it:
//   ./bin/tcyield --mode verify    --tasks 2000
//   ./bin/tcyield --mode calibrate                    # K sweep
//   ./bin/tcyield --mode overhead  --poll stage --tasks 20000 --reps 20
//   ./bin/tcyield --mode events    --discipline naive --events 10000
//                 [--oracle 1]

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
#include <cuda_fp16.h>
#include <mma.h>

#include "tile.cuh"  // flag contract, globaltimer, THREADS

using namespace nvcuda;

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err_ = (call);                                                \
    if (err_ != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err_));                                      \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

__device__ int* g_execCount;   // oracle counters: device globals, never
__device__ int* g_uExecCount;  // parameters (fault 5 / 18)

constexpr int BM = 128, BN = 128, BKT = 32, STAGES = 3;
constexpr int PADA = 8, PADB = 8;  // halves; keeps wmma ldm a multiple of 8
constexpr int LDA_S = BKT + PADA;  // 40
constexpr int LDB_S = BN + PADB;   // 136
constexpr size_t AS_STAGE = (size_t)BM * LDA_S;         // halves
constexpr size_t BS_STAGE = (size_t)BKT * LDB_S;        // halves
constexpr size_t SMEM_HALVES = STAGES * (AS_STAGE + BS_STAGE);
constexpr size_t SMEM_BYTES = SMEM_HALVES * sizeof(__half);  // 56,832 B

__device__ __forceinline__ void cp_async16(void* dst_smem, const void* src) {
  const unsigned s = (unsigned)__cvta_generic_to_shared(dst_smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s),
               "l"(src));
}
__device__ __forceinline__ void cp_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}
__device__ __forceinline__ void cp_wait_all() {
  asm volatile("cp.async.wait_all;\n" ::);
}
__device__ __forceinline__ void cp_wait2() {
  asm volatile("cp.async.wait_group 2;\n" ::);
}
__device__ __forceinline__ void cp_wait1() {
  asm volatile("cp.async.wait_group 1;\n" ::);
}

enum PollMode { POLL_OFF = 0, POLL_STAGE = 2, POLL_ISSUE = 3 };
enum Discipline { DISC_DRAIN = 0, DISC_NAIVE = 1, DISC_POISON = 2 };

template <int POLLMODE, int DISC, bool ORACLE = false>
__global__ void __launch_bounds__(THREADS) persistent_tc(
    const __half* __restrict__ A, const __half* __restrict__ B,
    float* __restrict__ C, float* __restrict__ CuScratch,
    const float* __restrict__ RefU, int* uBad, int K, int N, int tilesN,
    int totalTiles, unsigned numTasks, unsigned int* nextTask, int* flag,
    const long long* schedGapsNs, int numEvents, long long* setGT,
    long long* obsGT, int* claim, int* claimSite, long long* uGT) {
  extern __shared__ __half smem[];
  __shared__ unsigned s_task;
  __shared__ int s_flag;
  __shared__ int s_claimed;
  __shared__ int s_lastSeen;
  __shared__ int s_nextEv;
  __shared__ long long s_gBase;
  __shared__ int s_bad;

  const int warpId = (int)(threadIdx.x / 32);
  const int warpM = warpId % 4;  // 4 row-warps x 32 rows
  const int warpN = warpId / 4;  // 2 col-warps x 64 cols
  const bool isSetter = (schedGapsNs != nullptr) && blockIdx.x == 0;
  const int nChunks = K / BKT;

  if (threadIdx.x == 0) {
    s_flag = 0;
    s_lastSeen = 0;
    s_nextEv = 0;
    s_gBase = 0;
  }

  // cp.async coverage: per stage, As is 512 16-byte chunks and Bs is
  // 512; each thread moves 2 + 2.
  auto issue_chunk = [&](int st, int k0, int rowBase, int colBase) {
    __half* as = smem + (size_t)st * AS_STAGE;
    __half* bs = smem + (size_t)STAGES * AS_STAGE + (size_t)st * BS_STAGE;
    for (int i = 0; i < 2; ++i) {
      const int c = (int)threadIdx.x + i * THREADS;
      const int r = c / 4, col8 = (c % 4) * 8;
      cp_async16(as + (size_t)r * LDA_S + col8,
                 A + (size_t)(rowBase + r) * K + (k0 + col8));
    }
    for (int i = 0; i < 2; ++i) {
      const int c = (int)threadIdx.x + i * THREADS;
      const int r = c / 16, col8 = (c % 16) * 8;
      cp_async16(bs + (size_t)r * LDB_S + col8,
                 B + (size_t)(k0 + r) * N + (colBase + col8));
    }
    cp_commit();
  };

  using FragA = wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
                               wmma::row_major>;
  using FragB = wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
                               wmma::row_major>;
  using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

  auto compute_stage = [&](int st, FragC acc[2][4]) {
    __half* as = smem + (size_t)st * AS_STAGE;
    __half* bs = smem + (size_t)STAGES * AS_STAGE + (size_t)st * BS_STAGE;
#pragma unroll
    for (int kk = 0; kk < BKT; kk += 16) {
      FragA a[2];
      FragB b[4];
#pragma unroll
      for (int i = 0; i < 2; ++i)
        wmma::load_matrix_sync(
            a[i], as + (size_t)(warpM * 32 + i * 16) * LDA_S + kk, LDA_S);
#pragma unroll
      for (int j = 0; j < 4; ++j)
        wmma::load_matrix_sync(
            b[j], bs + (size_t)kk * LDB_S + warpN * 64 + j * 16, LDB_S);
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
          wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
    }
  };

  // Urgent tile: tile (0,0), staged synchronously through STAGE-0
  // buffers -- the smem in-flight cp.async groups may target.
  // Device-side verification against RefU; one atomicAdd of the bad
  // count per popping block.
  auto urgent_tile = [&](int event) {
    if (threadIdx.x == 0) {
      uGT[2 * (size_t)event] = globaltimer_ns();
      s_bad = 0;
    }
    FragC uacc[2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) wmma::fill_fragment(uacc[i][j], 0.0f);
    __half* as = smem;
    __half* bs = smem + (size_t)STAGES * AS_STAGE;
    for (int k0 = 0; k0 < K; k0 += BKT) {
      for (int c = (int)threadIdx.x; c < BM * BKT / 8; c += THREADS) {
        const int r = c / 4, col8 = (c % 4) * 8;
        for (int u = 0; u < 8; ++u)
          as[(size_t)r * LDA_S + col8 + u] =
              A[(size_t)r * K + (k0 + col8 + u)];
      }
      for (int c = (int)threadIdx.x; c < BKT * BN / 8; c += THREADS) {
        const int r = c / 16, col8 = (c % 16) * 8;
        for (int u = 0; u < 8; ++u)
          bs[(size_t)r * LDB_S + col8 + u] =
              B[(size_t)(k0 + r) * N + (col8 + u)];
      }
      if (DISC == DISC_POISON && threadIdx.x == 0 && k0 == 0)
        bs[0] = __hadd(bs[0], __float2half(1.0f));  // detector control
      __syncthreads();
      compute_stage(0, uacc);
      __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j)
        wmma::store_matrix_sync(
            CuScratch + (size_t)(warpM * 32 + i * 16) * BN + warpN * 64 +
                j * 16,
            uacc[i][j], BN, wmma::mem_row_major);
    __syncthreads();
    int myBad = 0;
    for (int e = (int)threadIdx.x; e < BM * BN; e += THREADS) {
      const float got = CuScratch[e];
      const float ref = RefU[e];
      const float err = fabsf(got - ref);
      if (err > 1e-3f + 1e-3f * fabsf(ref)) ++myBad;
    }
    atomicAdd(&s_bad, myBad);
    __syncthreads();
    if (threadIdx.x == 0) {
      if (s_bad > 0) atomicAdd(&uBad[event], s_bad);
      uGT[2 * (size_t)event + 1] = globaltimer_ns();
      if (ORACLE) atomicAdd(&g_uExecCount[event], 1);
    }
  };

  // Barrier-protected snapshot checkpoint (fault-8 discipline). site =
  // chunk index; -1 = boundary. Returns true when this block claimed
  // and must restart its tile.
  auto checkpoint = [&](int site, bool midPipeline) -> bool {
    const int f = s_flag;
    const int last = s_lastSeen;
    __syncthreads();
    if (f <= last) return false;
    const long long g = globaltimer_ns();
    if (threadIdx.x == 0) {
      for (int e = last; e < f; ++e)
        obsGT[(size_t)e * gridDim.x + blockIdx.x] = g;
      s_claimed = (atomicCAS(&claim[f - 1], 0, (int)blockIdx.x + 1) == 0);
      if (s_claimed) claimSite[f - 1] = site;
      s_lastSeen = f;
    }
    __syncthreads();
    if (!s_claimed) return false;
    if (midPipeline && (DISC == DISC_DRAIN || DISC == DISC_POISON))
      cp_wait_all();  // quiescence before smem reuse
    __syncthreads();
    urgent_tile(f - 1);
    cp_wait_all();  // stragglers must not leak into the restart
    __syncthreads();
    return true;
  };

  while (true) {
    if (threadIdx.x == 0) {
      s_task = atomicAdd(nextTask, 1u);
      if (POLLMODE != POLL_OFF) s_flag = flag_load(flag);
    }
    __syncthreads();
    const unsigned task = s_task;
    __syncthreads();
    if (task >= numTasks) break;

    const int tile = (int)(task % (unsigned)totalTiles);
    const int rowBase = (tile / tilesN) * BM;
    const int colBase = (tile % tilesN) * BN;

    bool redo = true;
    while (redo) {
      redo = false;
      FragC acc[2][4];
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

      issue_chunk(0, 0, rowBase, colBase);
      if (nChunks > 1) issue_chunk(1, BKT, rowBase, colBase);

      for (int c = 0; c < nChunks; ++c) {
        if (c + 2 < nChunks)
          issue_chunk((c + 2) % STAGES, (c + 2) * BKT, rowBase, colBase);

        if (POLLMODE == POLL_ISSUE) {
          if (threadIdx.x == 0) s_flag = flag_load(flag);
          __syncthreads();
          // Window-forced: before any wait, up to two copy groups are
          // outstanding, targeting stages (c+1)%3 and (c+2)%3 --
          // stage 0 (the urgent staging buffer) is in flight at two
          // of every three sites.
          if (checkpoint(c, true)) {
            redo = true;
            break;
          }
        }

        if (c + 2 < nChunks) cp_wait2();
        else if (c + 1 < nChunks) cp_wait1();
        else cp_wait_all();
        __syncthreads();

        if (isSetter && threadIdx.x == 0 && s_nextEv < numEvents) {
          if (s_gBase == 0) s_gBase = globaltimer_ns();
          const long long now = globaltimer_ns();
          if (now >= s_gBase + schedGapsNs[s_nextEv]) {
            setGT[s_nextEv] = now;
            flag_store(flag, ++s_nextEv);
          }
        }

        if (POLLMODE == POLL_STAGE) {
          if (threadIdx.x == 0) s_flag = flag_load(flag);
          __syncthreads();
          if (checkpoint(c, c + 1 < nChunks)) {
            redo = true;
            break;
          }
        }

        compute_stage(c % STAGES, acc);
        __syncthreads();
      }
      if (!redo) {
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
          for (int j = 0; j < 4; ++j)
            wmma::store_matrix_sync(
                C + (size_t)(rowBase + warpM * 32 + i * 16) * N + colBase +
                    warpN * 64 + j * 16,
                acc[i][j], N, wmma::mem_row_major);
        __syncthreads();
        if (ORACLE && threadIdx.x == 0) atomicAdd(&g_execCount[task], 1);
      }
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

static long long now_ns() {
  using namespace std::chrono;
  return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch())
      .count();
}

static double percentile(std::vector<double> v, double q) {
  std::sort(v.begin(), v.end());
  return v[std::min((size_t)(q * v.size()), v.size() - 1)];
}

int main(int argc, char** argv) {
  const std::string mode = parse_str(argc, argv, "--mode", "events");
  const std::string disc = parse_str(argc, argv, "--discipline", "drain");
  const std::string poll = parse_str(argc, argv, "--poll", "stage");
  const int K = (int)parse_arg(argc, argv, "--k", 256);
  const int tilesM = (int)parse_arg(argc, argv, "--tiles-m", 8);
  const int tilesN = (int)parse_arg(argc, argv, "--tiles-n", 8);
  const int events = (int)parse_arg(argc, argv, "--events", 10000);
  unsigned tasks = (unsigned)parse_arg(argc, argv, "--tasks", 20000);
  const int reps = (int)parse_arg(argc, argv, "--reps", 20);
  const double gapMinUs = parse_arg(argc, argv, "--gap-min-us", 150);
  const double gapMaxUs = parse_arg(argc, argv, "--gap-max-us", 400);
  int blocks = (int)parse_arg(argc, argv, "--blocks", 0);
  const int warmupEvents = (int)parse_arg(argc, argv, "--warmup-events", 100);
  const bool oracle = parse_arg(argc, argv, "--oracle", 0) != 0;
  const char* csvPath = parse_str(argc, argv, "--csv", "");

  if (K % BKT != 0 || K / BKT < STAGES) {
    fprintf(stderr, "--k must be a multiple of %d with >= %d chunks\n", BKT,
            STAGES);
    return 1;
  }
  const int discI = disc == "drain"    ? DISC_DRAIN
                    : disc == "naive"  ? DISC_NAIVE
                    : disc == "poison" ? DISC_POISON
                                       : -1;
  const int pollI = poll == "off"     ? POLL_OFF
                    : poll == "stage" ? POLL_STAGE
                    : poll == "issue" ? POLL_ISSUE
                                      : -1;
  if (discI < 0 || pollI < 0) {
    fprintf(stderr, "bad --discipline or --poll\n");
    return 1;
  }

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  const int sms = prop.multiProcessorCount;

  // Dynamic smem opt-in for every instantiation we may launch.
  auto allow = [&](const void* fn) {
    CUDA_CHECK(cudaFuncSetAttribute(
        fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES));
  };
  allow((const void*)persistent_tc<POLL_OFF, DISC_DRAIN>);
  allow((const void*)persistent_tc<POLL_STAGE, DISC_DRAIN>);
  allow((const void*)persistent_tc<POLL_STAGE, DISC_NAIVE>);
  allow((const void*)persistent_tc<POLL_STAGE, DISC_POISON>);
  allow((const void*)persistent_tc<POLL_ISSUE, DISC_NAIVE>);
  allow((const void*)persistent_tc<POLL_STAGE, DISC_DRAIN, true>);
  allow((const void*)persistent_tc<POLL_ISSUE, DISC_NAIVE, true>);

  int maxPerSM = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxPerSM, persistent_tc<POLL_STAGE, DISC_DRAIN>, THREADS,
      SMEM_BYTES));
  if (blocks <= 0) blocks = sms * maxPerSM;
  if (blocks > sms * maxPerSM) {
    fprintf(stderr, "ERROR: %d blocks > %d residency slots\n", blocks,
            sms * maxPerSM);
    return 1;
  }

  const int M = tilesM * BM;
  const int N = tilesN * BN;
  const int totalTiles = tilesM * tilesN;

  printf("GPU: %s | %d SMs | %d blocks/SM (smem %zu B) -> sat=%d | "
         "mode=%s disc=%s poll=%s | K=%d chunks=%d blocks=%d\n",
         prop.name, sms, maxPerSM, SMEM_BYTES, sms * maxPerSM, mode.c_str(),
         disc.c_str(), poll.c_str(), K, K / BKT, blocks);

  // fp16 inputs, deterministic; the float64 reference uses the
  // fp16-quantized values, so quantization is not part of the error.
  srand(21);
  std::vector<__half> hA((size_t)M * K), hB((size_t)K * N);
  std::vector<float> hAf(hA.size()), hBf(hB.size());
  for (size_t i = 0; i < hA.size(); ++i) {
    hAf[i] = (float)(rand() / (double)RAND_MAX * 2.0 - 1.0);
    hA[i] = __float2half(hAf[i]);
    hAf[i] = __half2float(hA[i]);
  }
  for (size_t i = 0; i < hB.size(); ++i) {
    hBf[i] = (float)(rand() / (double)RAND_MAX * 2.0 - 1.0);
    hB[i] = __float2half(hBf[i]);
    hBf[i] = __half2float(hB[i]);
  }

  // float64 reference for the urgent tile (tile 0), uploaded as float.
  std::vector<float> hRefU((size_t)BM * BN);
  for (int i = 0; i < BM; ++i)
    for (int j = 0; j < BN; ++j) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k)
        acc += (double)hAf[(size_t)i * K + k] * (double)hBf[(size_t)k * N + j];
      hRefU[(size_t)i * BN + j] = (float)acc;
    }

  __half *dA, *dB;
  float *dC, *dCuScratch, *dRefU;
  unsigned int* dNext;
  int *dFlag, *dClaim, *dClaimSite, *dUBad;
  long long *dSched, *dSetGT, *dObsGT, *dUGT;
  const int evAlloc = std::max(events, 1);
  CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dC, (size_t)M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dCuScratch, (size_t)BM * BN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dRefU, (size_t)BM * BN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dNext, sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc(&dFlag, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dClaim, (size_t)evAlloc * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dClaimSite, (size_t)evAlloc * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dUBad, (size_t)evAlloc * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dSched, (size_t)evAlloc * sizeof(long long)));
  CUDA_CHECK(cudaMalloc(&dSetGT, (size_t)evAlloc * sizeof(long long)));
  CUDA_CHECK(
      cudaMalloc(&dObsGT, (size_t)evAlloc * blocks * sizeof(long long)));
  CUDA_CHECK(cudaMalloc(&dUGT, (size_t)evAlloc * 2 * sizeof(long long)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(__half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(__half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dRefU, hRefU.data(),
                        (size_t)BM * BN * sizeof(float),
                        cudaMemcpyHostToDevice));

  cudaStream_t kStream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&kStream, cudaStreamNonBlocking));

  auto zero_all = [&]() {
    CUDA_CHECK(cudaMemset(dNext, 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(dFlag, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(dClaim, 0, (size_t)evAlloc * sizeof(int)));
    CUDA_CHECK(cudaMemset(dClaimSite, 0xFF, (size_t)evAlloc * sizeof(int)));
    CUDA_CHECK(cudaMemset(dUBad, 0, (size_t)evAlloc * sizeof(int)));
    CUDA_CHECK(cudaMemset(dSetGT, 0, (size_t)evAlloc * sizeof(long long)));
    CUDA_CHECK(
        cudaMemset(dObsGT, 0, (size_t)evAlloc * blocks * sizeof(long long)));
    CUDA_CHECK(cudaMemset(dUGT, 0, (size_t)evAlloc * 2 * sizeof(long long)));
  };

  auto launch = [&](int pm, int dc, unsigned nTasks, const long long* sched,
                    int nEvents) {
    auto args = [&](auto kern) {
      kern<<<blocks, THREADS, SMEM_BYTES, kStream>>>(
          dA, dB, dC, dCuScratch, dRefU, dUBad, K, N, tilesN, totalTiles,
          nTasks, dNext, dFlag, sched, nEvents, dSetGT, dObsGT, dClaim,
          dClaimSite, dUGT);
    };
    if (pm == POLL_OFF) args(persistent_tc<POLL_OFF, DISC_DRAIN>);
    else if (pm == POLL_STAGE && dc == DISC_DRAIN)
      args(persistent_tc<POLL_STAGE, DISC_DRAIN>);
    else if (pm == POLL_STAGE && dc == DISC_NAIVE)
      args(persistent_tc<POLL_STAGE, DISC_NAIVE>);
    else if (pm == POLL_STAGE)
      args(persistent_tc<POLL_STAGE, DISC_POISON>);
    else if (pm == POLL_ISSUE && dc == DISC_NAIVE)
      args(persistent_tc<POLL_ISSUE, DISC_NAIVE>);
    else {
      fprintf(stderr, "unsupported poll/discipline combination\n");
      exit(1);
    }
  };

  auto verify_C = [&](int coveredTiles) {
    std::vector<float> hC((size_t)M * N);
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    long long bad = 0, nans = 0;
    double maxRel = 0.0;
    const double rtol = 1e-3, atol = 1e-3;
    for (int t = 0; t < coveredTiles; ++t) {
      const int r0 = (t / tilesN) * BM, c0 = (t % tilesN) * BN;
      for (int i = 0; i < BM; ++i)
        for (int j = 0; j < BN; ++j) {
          double ref = 0.0;
          for (int k = 0; k < K; ++k)
            ref += (double)hAf[(size_t)(r0 + i) * K + k] *
                   (double)hBf[(size_t)k * N + (c0 + j)];
          const float got = hC[(size_t)(r0 + i) * N + (c0 + j)];
          if (std::isnan(got)) { ++nans; continue; }
          const double err = std::fabs((double)got - ref);
          maxRel = std::max(maxRel, err / std::max(1.0, std::fabs(ref)));
          if (err > atol + rtol * std::fabs(ref)) ++bad;
        }
    }
    printf("C check (%d tiles): %s — %lld NaN, %lld beyond tol, "
           "max rel %.3e\n",
           coveredTiles, (nans || bad) ? "FAIL" : "PASS", nans, bad, maxRel);
    return nans + bad == 0;
  };

  // ------------------------------------------------------------ verify
  if (mode == "verify") {
    const int vEvents = std::min(events, 5);
    std::vector<long long> gaps(evAlloc, 0);
    long long acc = 0;
    for (int e = 0; e < vEvents; ++e) gaps[e] = (acc += 400000);
    CUDA_CHECK(cudaMemcpy(dSched, gaps.data(),
                          (size_t)evAlloc * sizeof(long long),
                          cudaMemcpyHostToDevice));
    zero_all();
    CUDA_CHECK(cudaMemset(dC, 0xFF, (size_t)M * N * sizeof(float)));
    CUDA_CHECK(cudaDeviceSynchronize());
    launch(pollI == POLL_OFF ? POLL_STAGE : pollI, discI, tasks, dSched,
           vEvents);
    CUDA_CHECK(cudaStreamSynchronize(kStream));
    CUDA_CHECK(cudaGetLastError());

    const bool cOk = verify_C(std::min((int)tasks, totalTiles));
    std::vector<int> hClaimV(evAlloc), hBadV(evAlloc);
    CUDA_CHECK(cudaMemcpy(hClaimV.data(), dClaim, evAlloc * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hBadV.data(), dUBad, evAlloc * sizeof(int),
                          cudaMemcpyDeviceToHost));
    // host float64 confirmation of one captured urgent tile
    std::vector<float> hCu((size_t)BM * BN);
    CUDA_CHECK(cudaMemcpy(hCu.data(), dCuScratch,
                          (size_t)BM * BN * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double capMaxRel = 0.0;
    long long capBad = 0;
    for (int e = 0; e < BM * BN; ++e) {
      const double err = std::fabs((double)hCu[e] - (double)hRefU[e]);
      capMaxRel = std::max(capMaxRel,
                           err / std::max(1.0, std::fabs((double)hRefU[e])));
      if (err > 1e-3 + 1e-3 * std::fabs((double)hRefU[e])) ++capBad;
    }
    int claimed = 0, corrupt = 0;
    for (int e = 0; e < vEvents; ++e) {
      if (hClaimV[e]) ++claimed;
      if (hBadV[e] > 0) ++corrupt;
    }
    const bool expectCorrupt = (discI == DISC_POISON);
    printf("urgent: %d/%d claimed, %d device-flagged corrupt | captured "
           "tile host-check: %lld bad, max rel %.3e\n",
           claimed, vEvents, corrupt, capBad, capMaxRel);
    const bool pass =
        cOk && claimed > 0 &&
        (expectCorrupt ? (corrupt == claimed && capBad > 0)
                       : (corrupt == 0 && capBad == 0));
    printf("verify: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
  }

  // ---------------------------------------------------------- calibrate
  if (mode == "calibrate") {
    CUDA_CHECK(cudaMemset(dNext, 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(dFlag, 0, sizeof(int)));
    CUDA_CHECK(cudaDeviceSynchronize());
    const long long t0 = now_ns();
    launch(POLL_OFF, DISC_DRAIN, tasks, nullptr, 0);
    CUDA_CHECK(cudaStreamSynchronize(kStream));
    CUDA_CHECK(cudaGetLastError());
    const double wallUs = (now_ns() - t0) / 1e3;
    printf("calibrate K=%d: %u tasks on %d blocks in %.1f ms -> "
           "%.2f us/task/block | %.1f GFLOP/s\n",
           K, tasks, blocks, wallUs / 1e3, wallUs * blocks / tasks,
           2.0 * BM * BN * K * tasks / (wallUs * 1e3));
    return 0;
  }

  // ------------------------------------------------------------ overhead
  if (mode == "overhead") {
    const int warmPairs = 2;
    std::vector<double> wallOn, wallOff;
    for (int r = 0; r < 2 * (reps + warmPairs); ++r) {
      const bool on = ((r / 2) % 2 == 0) ? (r % 2 == 0) : (r % 2 == 1);
      CUDA_CHECK(cudaMemset(dNext, 0, sizeof(unsigned int)));
      CUDA_CHECK(cudaMemset(dFlag, 0, sizeof(int)));
      CUDA_CHECK(cudaDeviceSynchronize());
      const long long t0 = now_ns();
      launch(on ? pollI : POLL_OFF, DISC_DRAIN, tasks, nullptr, 0);
      CUDA_CHECK(cudaStreamSynchronize(kStream));
      CUDA_CHECK(cudaGetLastError());
      if (r >= 2 * warmPairs)
        (on ? wallOn : wallOff).push_back((now_ns() - t0) / 1e6);
    }
    const double on = percentile(wallOn, .5), off = percentile(wallOff, .5);
    printf("tc drain of %u tasks: poll %s %.3f ms | off %.3f ms | "
           "overhead %.2f%% | off-spread %.3f-%.3f\n",
           tasks, poll.c_str(), on, off, (on / off - 1.0) * 100.0,
           percentile(wallOff, 0.0), percentile(wallOff, 1.0));
    return 0;
  }

  // ------------------------------------------------------------ events
  {
    CUDA_CHECK(cudaMemset(dNext, 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(dFlag, 0, sizeof(int)));
    CUDA_CHECK(cudaDeviceSynchronize());
    const unsigned probeTasks = (unsigned)blocks * 40;
    const long long t0 = now_ns();
    launch(pollI, discI, probeTasks, nullptr, 0);
    CUDA_CHECK(cudaStreamSynchronize(kStream));
    const double usPerTask = (now_ns() - t0) / 1e3 / probeTasks;

    std::mt19937 rng(21);
    std::uniform_real_distribution<double> gapDist(gapMinUs * 1000.0,
                                                   gapMaxUs * 1000.0);
    std::vector<long long> gaps(evAlloc);
    long long accg = 0;
    for (int e = 0; e < events; ++e)
      gaps[e] = (accg += (long long)gapDist(rng));
    CUDA_CHECK(cudaMemcpy(dSched, gaps.data(),
                          (size_t)evAlloc * sizeof(long long),
                          cudaMemcpyHostToDevice));
    tasks = (unsigned)(1.5 * (accg / 1e3) / usPerTask);
    printf("probe %.3f us/task -> %u bounded tasks for %.0f ms schedule\n",
           usPerTask, tasks, accg / 1e6);

    int* dExecCount = nullptr;
    int* dUExec = nullptr;
    if (oracle) {
      CUDA_CHECK(cudaMalloc(&dExecCount, (size_t)tasks * sizeof(int)));
      CUDA_CHECK(cudaMalloc(&dUExec, (size_t)evAlloc * sizeof(int)));
      CUDA_CHECK(cudaMemset(dExecCount, 0, (size_t)tasks * sizeof(int)));
      CUDA_CHECK(cudaMemset(dUExec, 0, (size_t)evAlloc * sizeof(int)));
      CUDA_CHECK(
          cudaMemcpyToSymbol(g_execCount, &dExecCount, sizeof(int*)));
      CUDA_CHECK(cudaMemcpyToSymbol(g_uExecCount, &dUExec, sizeof(int*)));
    }
    zero_all();
    CUDA_CHECK(cudaMemset(dC, 0xFF, (size_t)M * N * sizeof(float)));
    CUDA_CHECK(cudaDeviceSynchronize());
    if (oracle) {
      auto args = [&](auto kern) {
        kern<<<blocks, THREADS, SMEM_BYTES, kStream>>>(
            dA, dB, dC, dCuScratch, dRefU, dUBad, K, N, tilesN, totalTiles,
            tasks, dNext, dFlag, dSched, events, dSetGT, dObsGT, dClaim,
            dClaimSite, dUGT);
      };
      if (pollI == POLL_STAGE && discI == DISC_DRAIN)
        args(persistent_tc<POLL_STAGE, DISC_DRAIN, true>);
      else if (pollI == POLL_ISSUE && discI == DISC_NAIVE)
        args(persistent_tc<POLL_ISSUE, DISC_NAIVE, true>);
      else {
        fprintf(stderr, "--oracle supports stage/drain, issue/naive\n");
        return 1;
      }
    } else {
      launch(pollI, discI, tasks, dSched, events);
    }
    CUDA_CHECK(cudaStreamSynchronize(kStream));
    CUDA_CHECK(cudaGetLastError());

    const bool cOk = verify_C(totalTiles);
    std::vector<long long> hSetGT(evAlloc), hUGT((size_t)evAlloc * 2),
        hObs((size_t)evAlloc * blocks);
    std::vector<int> hClaim(evAlloc), hSite(evAlloc), hBad(evAlloc);
    CUDA_CHECK(cudaMemcpy(hSetGT.data(), dSetGT, evAlloc * sizeof(long long),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hObs.data(), dObsGT,
                          (size_t)evAlloc * blocks * sizeof(long long),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hUGT.data(), dUGT,
                          (size_t)evAlloc * 2 * sizeof(long long),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hClaim.data(), dClaim, evAlloc * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hSite.data(), dClaimSite, evAlloc * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hBad.data(), dUBad, evAlloc * sizeof(int),
                          cudaMemcpyDeviceToHost));

    std::vector<double> lat, latX0, first, e2e;
    int anomalies = 0, unset = 0, unclaimed = 0, corrupt = 0,
        lastIsSetter = 0;
    FILE* csv = csvPath[0] ? fopen(csvPath, "w") : nullptr;
    if (csv)
      fprintf(csv, "event,lat_us,lat_excl_setter_us,first_us,claimer,"
                   "e2e_us,bad_elems,site,complete\n");
    for (int e = 0; e < events; ++e) {
      const long long setNs = hSetGT[e];
      if (setNs == 0) { ++unset; continue; }
      long long lo = 0, hi = 0, hiX0 = 0;
      int seen = 0, lastBlock = -1;
      for (int b = 0; b < blocks; ++b) {
        const long long o = hObs[(size_t)e * blocks + b];
        if (o == 0) continue;
        ++seen;
        lo = (lo == 0 || o < lo) ? o : lo;
        if (o > hi) { hi = o; lastBlock = b; }
        if (b > 0 && o > hiX0) hiX0 = o;
      }
      const bool complete = (seen == blocks);
      if (!complete) ++anomalies;
      if (!hClaim[e]) ++unclaimed;
      if (hBad[e] > 0) ++corrupt;
      const double latUs = (hi - setNs) / 1e3;
      const double e2eUs =
          hClaim[e] ? (hUGT[2 * (size_t)e + 1] - setNs) / 1e3 : -1.0;
      if (complete && e >= warmupEvents) {
        lat.push_back(latUs);
        latX0.push_back((hiX0 - setNs) / 1e3);
        first.push_back((lo - setNs) / 1e3);
        if (lastBlock == 0) ++lastIsSetter;
        if (e2eUs >= 0) e2e.push_back(e2eUs);
      }
      if (csv)
        fprintf(csv, "%d,%.2f,%.2f,%.2f,%d,%.2f,%d,%d,%d\n", e, latUs,
                (hiX0 - setNs) / 1e3, (lo - setNs) / 1e3, hClaim[e] - 1,
                e2eUs, hBad[e], hSite[e], complete ? 1 : 0);
    }
    if (csv) fclose(csv);

    const int tested = events - unset - unclaimed;
    printf("\n--- tc %s/%s: %d events (%d unset, %d unclaimed, %d "
           "anomalies) ---\n",
           disc.c_str(), poll.c_str(), events, unset, unclaimed, anomalies);
    printf("SAFETY: %d/%d urgent tiles corrupt | bg C %s\n", corrupt,
           tested, cOk ? "PASS" : "FAIL");
    if (!lat.empty()) {
      printf("latency set->last (us): p50 %.2f | p90 %.2f | p99 %.2f  "
             "(excl setter p50 %.2f, setter-last %.1f%%)\n",
             percentile(lat, .5), percentile(lat, .9), percentile(lat, .99),
             percentile(latX0, .5), 100.0 * lastIsSetter / lat.size());
      printf("first obs p50 %.2f | urgent e2e p50 %.2f p99 %.2f\n",
             percentile(first, .5), e2e.empty() ? -1 : percentile(e2e, .5),
             e2e.empty() ? -1 : percentile(e2e, .99));
    }
    if (csvPath[0]) printf("raw: %s\n", csvPath);

    if (oracle) {
      std::vector<int> hExec(tasks), hUEx(evAlloc);
      CUDA_CHECK(cudaMemcpy(hExec.data(), dExecCount,
                            (size_t)tasks * sizeof(int),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(hUEx.data(), dUExec, evAlloc * sizeof(int),
                            cudaMemcpyDeviceToHost));
      long long dup = 0, missed = 0;
      for (unsigned t = 0; t < tasks; ++t)
        if (hExec[t] != 1) { if (hExec[t] > 1) ++dup; else ++missed; }
      int fired = 0, badU = 0;
      for (int e = 0; e < events; ++e) {
        if (hSetGT[e] == 0) continue;
        ++fired;
        if (hUEx[e] != 1) ++badU;
      }
      printf("oracle tasks: %u | dup %lld | missed %lld\n", tasks, dup,
             missed);
      printf("oracle events: %d fired | urgent violations %d\n", fired,
             badU);
      const bool pass =
          dup == 0 && missed == 0 && badU == 0 && fired == events;
      printf("ORACLE: %s\n", pass ? "PASS" : "FAIL");
      return pass ? 0 : 1;
    }
    return 0;
  }
}
