// Phase 5 — the safety question: is per-block independent yield sound on
// sm_86 when the yield lands MID-PIPELINE, and under what conditions?
//
// ExpertPlex: "independent tile-boundary checks are unsafe because
// high-performance operations pipeline both warps and CTAs across tiles."
// That is a statement about Hopper, where TMA multicast and DSMEM couple
// CTAs in a cluster. sm_86 has no cluster coupling — but it does have the
// warp-level half of the hazard: cp.async copies in flight into shared
// memory. This file builds a genuinely pipelined tile (double-buffered
// cp.async staging, one commit group per chunk) and forces the claimer to
// interrupt it mid-flight, then reuses the same shared memory for the
// urgent tile under one of three disciplines:
//
//   drain   cp.async.wait_all + barrier before the smem is reused.
//           Hypothesis: always correct -> independent yield is safe on
//           sm_86 given one wait instruction.
//   naive   no wait: the outstanding copy group races the urgent staging.
//           Hypothesis: measurable corruption of the urgent output — the
//           ExpertPlex hazard class reproduced inside one CTA.
//   poison  drain, then deliberately corrupt one staged float. Must fail
//           100% of events — proves the detector detects (the Cornfield
//           negative-control rule: no null result without a nonzero
//           instrument reading).
//
// The abandoned background tile RESTARTS from scratch after the yield, so
// C stays correct by construction in every discipline; the safety verdict
// lives in the per-event urgent outputs. All of them are captured
// (CuAll[event]) and checked against a float64 host reference — never
// against another run of the same kernel (CLAIM.md).
//
// Run it (scripts/phase5.sh drives the full set):
//   ./bin/midyield --mode verify   --tiles-m 8 --tiles-n 8 --tasks 3000
//   ./bin/midyield --mode events   --discipline naive --events 10000
//   ./bin/midyield --mode overhead --poll stage --tasks 200000 --reps 20
//
// Instrument notes carried from Phases 3-4: %globaltimer ticks at
// 1.024 us; occupancy pinned to 4/SM via launch bounds; overhead compares
// are within-binary only (codegen caveat); the dev-set setter is block 0
// and the analysis reports latency with and without it.

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

// Pipelined-tile geometry. KBP-deep chunks, two smem stages, so K=64 runs
// a 4-chunk pipeline. 256 threads move one 16 B cp.async each for As and
// Bs per chunk (64x16 + 16x64 floats = 2 x 256 x 16 B). No smem padding:
// cp.async needs 16 B-aligned rows, and this file measures safety, not
// bank conflicts.
constexpr int KBP = 16;

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
__device__ __forceinline__ void cp_wait_one_outstanding() {
  asm volatile("cp.async.wait_group 1;\n" ::);
}

enum PollMode { POLL_OFF = 0, POLL_BOUNDARY = 1, POLL_STAGE = 2 };
enum Discipline { DISC_DRAIN = 0, DISC_NAIVE = 1, DISC_POISON = 2 };

template <int POLLMODE, int DISC>
__global__ void __launch_bounds__(THREADS, 4) persistent_midyield(
    const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, float* __restrict__ CuAll, int K, int N,
    int tilesN, int totalTiles, unsigned numTasks, unsigned int* nextTask,
    volatile int* flag, const long long* schedGapsNs, int numEvents,
    long long* setGT, long long* obsGT, int* claim, long long* uGT) {
  __shared__ float As[2][TM][KBP];
  __shared__ float Bs[2][KBP][TN];
  __shared__ unsigned s_task;
  __shared__ int s_flag;
  __shared__ int s_claimed;
  __shared__ int s_lastSeen;
  __shared__ int s_nextEv;
  __shared__ long long s_gBase;

  const int tx = threadIdx.x % 16;
  const int ty = threadIdx.x / 16;
  const bool isSetter = (schedGapsNs != nullptr) && blockIdx.x == 0;
  const int nChunks = K / KBP;

  if (threadIdx.x == 0) {
    s_flag = 0;
    s_lastSeen = 0;
    s_nextEv = 0;
    s_gBase = 0;
  }

  // One cp.async pair per thread per chunk: thread t stages As row t/4,
  // 16 B column chunk t%4, and Bs row t/16, 16 B column chunk t%16.
  const int arow = threadIdx.x / 4, acol = (threadIdx.x % 4) * 4;
  const int brow = threadIdx.x / 16, bcol = (threadIdx.x % 16) * 4;

  auto issue_chunk = [&](int buf, int k0, int rowBase, int colBase) {
    cp_async16(&As[buf][arow][acol],
               &A[(long long)(rowBase + arow) * K + (k0 + acol)]);
    cp_async16(&Bs[buf][brow][bcol],
               &B[(long long)(k0 + brow) * N + (colBase + bcol)]);
    cp_commit();
  };

  // The urgent tile, staged synchronously through buffer 0 — the same
  // smem an outstanding cp.async group may still be landing into. That
  // collision is the experiment; the discipline decides whether it can
  // happen. Computes tile (0,0) into CuAll[event].
  auto urgent_tile = [&](int event) {
    if (threadIdx.x == 0) uGT[2 * (size_t)event] = globaltimer_ns();
    float uacc[4][4] = {};
    for (int k0 = 0; k0 < K; k0 += KBP) {
      for (int i = 0; i < 4; ++i) {
        As[0][arow][acol + i] = A[(long long)arow * K + (k0 + acol + i)];
        Bs[0][brow][bcol + i] = B[(long long)(k0 + brow) * N + (bcol + i)];
      }
      if (DISC == DISC_POISON && threadIdx.x == 0 && k0 == 0)
        Bs[0][0][0] += 1.0f;  // detector control: one wrong staged value
      __syncthreads();
#pragma unroll
      for (int kk = 0; kk < KBP; ++kk) {
        float a[4], b[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) a[i] = As[0][ty * 4 + i][kk];
#pragma unroll
        for (int j = 0; j < 4; ++j) b[j] = Bs[0][kk][tx * 4 + j];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
          for (int j = 0; j < 4; ++j)
            uacc[i][j] = fmaf(a[i], b[j], uacc[i][j]);
      }
      __syncthreads();
    }
    float* Cu = CuAll + (size_t)event * TM * TN;
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j)
        Cu[(ty * 4 + i) * TN + (tx * 4 + j)] = uacc[i][j];
    __syncthreads();
    if (threadIdx.x == 0) uGT[2 * (size_t)event + 1] = globaltimer_ns();
  };

  // Event checkpoint. Returns true (uniformly) when this block claimed the
  // event and must abandon + restart its tile. Non-claimers only log.
  auto checkpoint = [&](bool midPipeline) -> bool {
    if (s_flag <= s_lastSeen) return false;
    const long long g = globaltimer_ns();
    const int f = s_flag;
    if (threadIdx.x == 0) {
      for (int e = s_lastSeen; e < f; ++e)
        obsGT[(size_t)e * gridDim.x + blockIdx.x] = g;
      s_claimed = (atomicCAS(&claim[f - 1], 0, (int)blockIdx.x + 1) == 0);
      s_lastSeen = f;
    }
    __syncthreads();
    if (!s_claimed) return false;
    if (midPipeline && (DISC == DISC_DRAIN || DISC == DISC_POISON))
      cp_wait_all();  // the one instruction the drain discipline adds
    __syncthreads();
    urgent_tile(f - 1);
    cp_wait_all();  // naive cleanup: stragglers must not leak into restart
    __syncthreads();
    return true;
  };

  while (true) {
    if (threadIdx.x == 0) {
      s_task = atomicAdd(nextTask, 1u);
      if (POLLMODE != POLL_OFF) s_flag = *flag;
    }
    __syncthreads();
    const unsigned task = s_task;
    __syncthreads();
    if (task >= numTasks) break;

    const int tile = (int)(task % (unsigned)totalTiles);
    const int rowBase = (tile / tilesN) * TM;
    const int colBase = (tile % tilesN) * TN;

    if (POLLMODE == POLL_BOUNDARY) checkpoint(false);  // pipeline is empty

    bool redo = true;
    while (redo) {
      redo = false;
      float acc[4][4] = {};
      issue_chunk(0, 0, rowBase, colBase);
      for (int c = 0; c < nChunks; ++c) {
        if (c + 1 < nChunks)
          issue_chunk((c + 1) & 1, (c + 1) * KBP, rowBase, colBase);
        if (c + 1 < nChunks)
          cp_wait_one_outstanding();  // chunk c landed, c+1 in flight
        else
          cp_wait_all();
        __syncthreads();

        if (isSetter && threadIdx.x == 0 && s_nextEv < numEvents) {
          if (s_gBase == 0) s_gBase = globaltimer_ns();
          const long long now = globaltimer_ns();
          if (now >= s_gBase + schedGapsNs[s_nextEv]) {
            setGT[s_nextEv] = now;
            __threadfence();
            *flag = ++s_nextEv;
          }
        }

        if (POLLMODE == POLL_STAGE) {
          if (threadIdx.x == 0) s_flag = *flag;
          __syncthreads();
          // Mid-pipeline yield point: for c+1 < nChunks a copy group is
          // still outstanding, targeting buffer (c+1)&1 — buffer 0 on
          // every second chunk, i.e. exactly where urgent_tile stages.
          if (checkpoint(c + 1 < nChunks)) {
            redo = true;
            break;
          }
        }

#pragma unroll
        for (int kk = 0; kk < KBP; ++kk) {
          float a[4], b[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) a[i] = As[c & 1][ty * 4 + i][kk];
#pragma unroll
          for (int j = 0; j < 4; ++j) b[j] = Bs[c & 1][kk][tx * 4 + j];
#pragma unroll
          for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j)
              acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
      }
      if (!redo) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const long long r = rowBase + ty * 4 + i;
#pragma unroll
          for (int j = 0; j < 4; ++j)
            C[r * N + (colBase + tx * 4 + j)] = acc[i][j];
        }
        __syncthreads();
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
  const int K = (int)parse_arg(argc, argv, "--k", 64);
  const int tilesM = (int)parse_arg(argc, argv, "--tiles-m", 48);
  const int tilesN = (int)parse_arg(argc, argv, "--tiles-n", 48);
  const int events = (int)parse_arg(argc, argv, "--events", 10000);
  unsigned tasks = (unsigned)parse_arg(argc, argv, "--tasks", 200000);
  const int reps = (int)parse_arg(argc, argv, "--reps", 20);
  const double gapMinUs = parse_arg(argc, argv, "--gap-min-us", 150);
  const double gapMaxUs = parse_arg(argc, argv, "--gap-max-us", 400);
  int blocks = (int)parse_arg(argc, argv, "--blocks", 0);
  const int warmupEvents = (int)parse_arg(argc, argv, "--warmup-events", 100);
  const char* csvPath = parse_str(argc, argv, "--csv", "");

  if (K % KBP != 0 || (K / KBP) < 2) {
    fprintf(stderr, "--k must be a multiple of %d with >= 2 chunks\n", KBP);
    return 1;
  }
  const int discI = disc == "drain"    ? DISC_DRAIN
                    : disc == "naive"  ? DISC_NAIVE
                    : disc == "poison" ? DISC_POISON
                                       : -1;
  const int pollI = poll == "off"        ? POLL_OFF
                    : poll == "boundary" ? POLL_BOUNDARY
                    : poll == "stage"    ? POLL_STAGE
                                         : -1;
  if (discI < 0 || pollI < 0) {
    fprintf(stderr, "bad --discipline or --poll\n");
    return 1;
  }

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  const int sms = prop.multiProcessorCount;
  int maxPerSM = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxPerSM, persistent_midyield<POLL_STAGE, DISC_DRAIN>, THREADS, 0));
  if (blocks <= 0) blocks = sms * maxPerSM;
  if (blocks > sms * maxPerSM) {
    fprintf(stderr, "ERROR: %d blocks > %d residency slots\n", blocks,
            sms * maxPerSM);
    return 1;
  }

  const int M = tilesM * TM;
  const int N = tilesN * TN;
  const int totalTiles = tilesM * tilesN;

  printf("GPU: %s | %d SMs | %d blocks/SM -> sat=%d | mode=%s disc=%s "
         "poll=%s | K=%d chunks=%d blocks=%d\n",
         prop.name, sms, maxPerSM, sms * maxPerSM, mode.c_str(), disc.c_str(),
         poll.c_str(), K, K / KBP, blocks);

  srand(21);
  std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
  for (auto& v : hA) v = (float)(rand() / (double)RAND_MAX * 2.0 - 1.0);
  for (auto& v : hB) v = (float)(rand() / (double)RAND_MAX * 2.0 - 1.0);

  const int evAlloc = std::max(events, 1);
  float *dA, *dB, *dC, *dCuAll;
  unsigned int* dNext;
  int *dFlag, *dClaim;
  long long *dSched, *dSetGT, *dObsGT, *dUGT;
  CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, (size_t)M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dCuAll, (size_t)evAlloc * TM * TN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dNext, sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc(&dFlag, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dSched, (size_t)evAlloc * sizeof(long long)));
  CUDA_CHECK(cudaMalloc(&dSetGT, (size_t)evAlloc * sizeof(long long)));
  CUDA_CHECK(cudaMalloc(&dObsGT, (size_t)evAlloc * blocks * sizeof(long long)));
  CUDA_CHECK(cudaMalloc(&dUGT, (size_t)evAlloc * 2 * sizeof(long long)));
  CUDA_CHECK(cudaMalloc(&dClaim, (size_t)evAlloc * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  cudaStream_t kStream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&kStream, cudaStreamNonBlocking));

  auto zero_all = [&]() {
    CUDA_CHECK(cudaMemset(dNext, 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(dFlag, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(dSetGT, 0, (size_t)evAlloc * sizeof(long long)));
    CUDA_CHECK(
        cudaMemset(dObsGT, 0, (size_t)evAlloc * blocks * sizeof(long long)));
    CUDA_CHECK(cudaMemset(dUGT, 0, (size_t)evAlloc * 2 * sizeof(long long)));
    CUDA_CHECK(cudaMemset(dClaim, 0, (size_t)evAlloc * sizeof(int)));
  };

  auto launch = [&](int pm, int dc, unsigned nTasks, const long long* sched,
                    int nEvents) {
    auto args = [&](auto kern) {
      kern<<<blocks, THREADS, 0, kStream>>>(dA, dB, dC, dCuAll, K, N, tilesN,
                                            totalTiles, nTasks, dNext, dFlag,
                                            sched, nEvents, dSetGT, dObsGT,
                                            dClaim, dUGT);
    };
    if (pm == POLL_OFF) args(persistent_midyield<POLL_OFF, DISC_DRAIN>);
    else if (pm == POLL_BOUNDARY)
      args(persistent_midyield<POLL_BOUNDARY, DISC_DRAIN>);
    else if (dc == DISC_DRAIN)
      args(persistent_midyield<POLL_STAGE, DISC_DRAIN>);
    else if (dc == DISC_NAIVE)
      args(persistent_midyield<POLL_STAGE, DISC_NAIVE>);
    else
      args(persistent_midyield<POLL_STAGE, DISC_POISON>);
  };

  // float64 reference for the urgent tile — tile (0,0), computed once.
  std::vector<double> refU((size_t)TM * TN, 0.0);
  for (int i = 0; i < TM; ++i)
    for (int k = 0; k < K; ++k) {
      const double a = hA[(size_t)i * K + k];
      for (int j = 0; j < TN; ++j) refU[(size_t)i * TN + j] += a * hB[(size_t)k * N + j];
    }
  const double rtol = 1e-4, atol = 1e-4;
  auto check_cu = [&](const float* cu, double& maxRel) {
    int bad = 0;
    for (int i = 0; i < TM * TN; ++i) {
      if (std::isnan(cu[i])) { ++bad; continue; }
      const double err = std::fabs(cu[i] - refU[i]);
      maxRel = std::max(maxRel, err / std::max(1.0, std::fabs(refU[i])));
      if (err > atol + rtol * std::fabs(refU[i])) ++bad;
    }
    return bad;
  };
  auto verify_C = [&](int coveredTiles) {
    std::vector<float> hC((size_t)M * N);
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    int bad = 0, nans = 0;
    double maxRel = 0.0;
    for (int tile = 0; tile < coveredTiles; ++tile) {
      const int r0 = (tile / tilesN) * TM, c0 = (tile % tilesN) * TN;
      for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) {
          double ref = 0.0;
          for (int k = 0; k < K; ++k)
            ref += (double)hA[(size_t)(r0 + i) * K + k] *
                   hB[(size_t)k * N + (c0 + j)];
          const float got = hC[(size_t)(r0 + i) * N + (c0 + j)];
          if (std::isnan(got)) { ++nans; continue; }
          const double err = std::fabs(got - ref);
          maxRel = std::max(maxRel, err / std::max(1.0, std::fabs(ref)));
          if (err > atol + rtol * std::fabs(ref)) ++bad;
        }
    }
    printf("C check (%d tiles): %s — %d NaN, %d beyond tol, max rel %.3e\n",
           coveredTiles, (nans || bad) ? "FAIL" : "PASS", nans, bad, maxRel);
    return nans + bad == 0;
  };

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
      launch(on ? pollI : POLL_OFF, discI, tasks, nullptr, 0);
      CUDA_CHECK(cudaStreamSynchronize(kStream));
      CUDA_CHECK(cudaGetLastError());
      if (r >= 2 * warmPairs)
        (on ? wallOn : wallOff).push_back((now_ns() - t0) / 1e6);
    }
    const double on = percentile(wallOn, 0.5), off = percentile(wallOff, 0.5);
    printf("pipelined-tile drain of %u tasks, %d reps (order-alternated):\n",
           tasks, reps);
    printf("  poll %s: %.3f ms | poll off: %.3f ms | overhead %.2f%%\n",
           poll.c_str(), on, off, (on / off - 1.0) * 100.0);
    return 0;
  }

  // ------------------------------------------------------------ verify
  if (mode == "verify") {
    const int vEvents = std::min(events, 5);
    std::vector<long long> gaps(evAlloc, 0);
    long long acc = 0;
    for (int e = 0; e < vEvents; ++e) gaps[e] = (acc += 300000);
    CUDA_CHECK(cudaMemcpy(dSched, gaps.data(),
                          (size_t)evAlloc * sizeof(long long),
                          cudaMemcpyHostToDevice));
    zero_all();
    CUDA_CHECK(cudaMemset(dC, 0xFF, (size_t)M * N * sizeof(float)));
    CUDA_CHECK(
        cudaMemset(dCuAll, 0xFF, (size_t)evAlloc * TM * TN * sizeof(float)));
    CUDA_CHECK(cudaDeviceSynchronize());
    launch(pollI, discI, tasks, dSched, vEvents);
    CUDA_CHECK(cudaStreamSynchronize(kStream));
    CUDA_CHECK(cudaGetLastError());

    bool ok = verify_C(std::min((int)tasks, totalTiles));
    std::vector<int> hClaim(evAlloc);
    CUDA_CHECK(cudaMemcpy(hClaim.data(), dClaim, evAlloc * sizeof(int),
                          cudaMemcpyDeviceToHost));
    std::vector<float> hCu((size_t)evAlloc * TM * TN);
    CUDA_CHECK(cudaMemcpy(hCu.data(), dCuAll, hCu.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    int claimed = 0, corrupt = 0;
    for (int e = 0; e < vEvents; ++e) {
      if (!hClaim[e]) continue;
      ++claimed;
      double mr = 0.0;
      if (check_cu(&hCu[(size_t)e * TM * TN], mr) > 0) ++corrupt;
    }
    printf("urgent tiles: %d/%d claimed, %d corrupt (discipline=%s)\n",
           claimed, vEvents, corrupt, disc.c_str());
    const bool expectCorrupt = (discI == DISC_POISON);
    const bool pass = ok && claimed > 0 &&
                      (expectCorrupt ? corrupt == claimed : corrupt == 0);
    printf("verify: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
  }

  // ------------------------------------------------------------ events
  // Size the background queue off a probe drain so the schedule fits with
  // ~1.5x margin, rather than guessing.
  {
    CUDA_CHECK(cudaMemset(dNext, 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(dFlag, 0, sizeof(int)));
    CUDA_CHECK(cudaDeviceSynchronize());
    const unsigned probeTasks = (unsigned)blocks * 100;
    const long long t0 = now_ns();
    launch(pollI, discI, probeTasks, nullptr, 0);
    CUDA_CHECK(cudaStreamSynchronize(kStream));
    const double usPerTask = (now_ns() - t0) / 1e3 / probeTasks;
    const double schedUs = events * (gapMinUs + gapMaxUs) / 2.0;
    tasks = (unsigned)(1.5 * schedUs / usPerTask);
    printf("probe: %.3f us/task -> %u bg tasks for a %.1f ms schedule\n",
           usPerTask, tasks, schedUs / 1e3);
  }

  std::mt19937 rng(21);
  std::uniform_real_distribution<double> gapDist(gapMinUs * 1000.0,
                                                 gapMaxUs * 1000.0);
  std::vector<long long> gaps(evAlloc);
  long long accg = 0;
  for (int e = 0; e < events; ++e) gaps[e] = (accg += (long long)gapDist(rng));
  CUDA_CHECK(cudaMemcpy(dSched, gaps.data(),
                        (size_t)evAlloc * sizeof(long long),
                        cudaMemcpyHostToDevice));
  zero_all();
  CUDA_CHECK(cudaMemset(dC, 0xFF, (size_t)M * N * sizeof(float)));
  CUDA_CHECK(
      cudaMemset(dCuAll, 0xFF, (size_t)evAlloc * TM * TN * sizeof(float)));
  CUDA_CHECK(cudaDeviceSynchronize());
  launch(pollI, discI, tasks, dSched, events);
  CUDA_CHECK(cudaStreamSynchronize(kStream));
  CUDA_CHECK(cudaGetLastError());

  std::vector<long long> hSetGT(evAlloc), hUGT((size_t)evAlloc * 2);
  std::vector<long long> hObs((size_t)evAlloc * blocks);
  std::vector<int> hClaim(evAlloc);
  std::vector<float> hCu((size_t)evAlloc * TM * TN);
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
  CUDA_CHECK(cudaMemcpy(hCu.data(), dCuAll, hCu.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  const bool cOk = verify_C(totalTiles);

  std::vector<double> lat, latX0, first, e2e;
  int anomalies = 0, unset = 0, unclaimed = 0, corrupt = 0, lastIsSetter = 0;
  double worstRel = 0.0;
  FILE* csv = csvPath[0] ? fopen(csvPath, "w") : nullptr;
  if (csv)
    fprintf(csv, "event,lat_us,lat_excl_setter_us,first_us,claimer,e2e_us,"
                 "cu_max_rel,corrupt,complete\n");
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
    double mr = 0.0;
    int bad = 0;
    if (hClaim[e]) {
      bad = check_cu(&hCu[(size_t)e * TM * TN], mr);
      worstRel = std::max(worstRel, mr);
      if (bad > 0) ++corrupt;
    } else {
      ++unclaimed;
    }
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
      fprintf(csv, "%d,%.2f,%.2f,%.2f,%d,%.2f,%.3e,%d,%d\n", e, latUs,
              (hiX0 - setNs) / 1e3, (lo - setNs) / 1e3, hClaim[e] - 1, e2eUs,
              mr, bad > 0 ? 1 : 0, complete ? 1 : 0);
  }
  if (csv) fclose(csv);

  const int tested = events - unset - unclaimed;
  printf("\n--- %s / %s: %d events (%d unset, %d unclaimed, %d anomalies) "
         "---\n",
         disc.c_str(), poll.c_str(), events, unset, unclaimed, anomalies);
  printf("SAFETY: %d/%d urgent tiles corrupt (%.2f%%) | worst rel err "
         "%.3e | bg C %s\n",
         corrupt, tested, 100.0 * corrupt / std::max(tested, 1), worstRel,
         cOk ? "PASS" : "FAIL");
  if (!lat.empty()) {
    printf("latency set->last (us): p50 %.2f | p90 %.2f | p99 %.2f  "
           "(excl setter p50 %.2f, setter-last %.1f%%)\n",
           percentile(lat, 0.5), percentile(lat, 0.9), percentile(lat, 0.99),
           percentile(latX0, 0.5), 100.0 * lastIsSetter / lat.size());
    printf("first observer p50 %.2f us | urgent e2e p50 %.2f us\n",
           percentile(first, 0.5), e2e.empty() ? -1 : percentile(e2e, 0.5));
  }
  if (csvPath[0]) printf("raw: %s\n", csvPath);
  return 0;
}
