// Phase 3 — the naive yield, and the two numbers (ROADMAP Phase 3).
//
// Mechanism under test: a global device-memory flag, each block polling
// INDEPENDENTLY at its own task boundary (exactly what ExpertPlex calls
// unsafe on Hopper; safety is Phase 5 — here we measure cost and latency).
//
// The two numbers:
//   1. Preemption latency: flag set -> last resident worker at a yield
//      point (CLAIM.md definition). p99 over >= 10k events.
//   2. Steady-state overhead: throughput with polling on vs off, same
//      shape, no events. CLAIM.md falsifies N1 above 10%.
//
// Timestamping the set-instant is the instrument problem. Two set modes,
// which must agree up to PCIe visibility:
//   --set dev   block 0 sets the flag at pre-scheduled instants and logs
//               %globaltimer at the write. Same clock as observers, zero
//               cross-domain error. The primary number.
//   --set host  the host sets the flag (CLAIM.md's literal wording) via
//               cudaMemcpyAsync; host steady_clock is mapped onto
//               %globaltimer with a spin-kernel calibration (offset
//               distribution reported; measured idle — stated limitation).
//
// Per event, every block logs its observation time; the claimer (atomicCAS)
// executes the urgent tile so e2e is comparable with Phase 2's 957 us
// baseline. Blocks then resume the background queue; no work is lost (the
// popped task is executed after the yield point, not dropped).
//
// One long-running kernel hosts all 10k events (gaps >> latency so events
// never overlap); anomalies (a block missing an event) are counted, not
// silently blended.

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

template <bool POLL>
__global__ void __launch_bounds__(THREADS) persistent_yield(
    const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, float* __restrict__ Cu, int K, int N, int tilesN,
    int totalTiles, unsigned numTasks, unsigned int* nextTask,
    volatile int* flag, const long long* schedGapsNs, int numEvents,
    long long* setGT, long long* obsGT, int* claim, long long* uGT) {
  __shared__ unsigned s_task;
  __shared__ int s_flag;
  __shared__ int s_claimed;

  int lastSeen = 0;
  long long gBase = 0;
  int nextEv = 0;
  const bool isSetter = (schedGapsNs != nullptr) && blockIdx.x == 0;

  while (true) {
    if (threadIdx.x == 0) {
      s_task = atomicAdd(nextTask, 1u);
      if (POLL) s_flag = *flag;  // one volatile L2 read per tile boundary
    }
    __syncthreads();
    const unsigned task = s_task;
    const int f = POLL ? s_flag : 0;
    __syncthreads();
    if (f < 0 || task >= numTasks) break;  // -1 = host stop sentinel

    if (POLL && f > lastSeen) {
      // Yield point: this block is redirectable NOW (previous tile done,
      // popped task not yet started). Observation time is the "switched"
      // timestamp in the preemption-latency definition.
      const long long g = globaltimer_ns();
      if (threadIdx.x == 0) {
        for (int e = lastSeen; e < f; ++e)  // f-lastSeen>1 = missed event(s)
          obsGT[(size_t)e * gridDim.x + blockIdx.x] = g;
        s_claimed = (atomicCAS(&claim[f - 1], 0, (int)blockIdx.x + 1) == 0);
      }
      lastSeen = f;
      __syncthreads();
      if (s_claimed) {
        if (threadIdx.x == 0) uGT[2 * (size_t)(f - 1)] = globaltimer_ns();
        tile_gemm(A, B, Cu, K, N, 0, 0, TN, 0, 0);
        __syncthreads();
        if (threadIdx.x == 0) uGT[2 * (size_t)(f - 1) + 1] = globaltimer_ns();
      }
    }

    if (isSetter && threadIdx.x == 0 && nextEv < numEvents) {
      if (gBase == 0) gBase = globaltimer_ns();
      const long long now = globaltimer_ns();
      if (now >= gBase + schedGapsNs[nextEv]) {
        setGT[nextEv] = now;
        __threadfence();  // setGT visible before the flag flips
        *flag = ++nextEv;
      }
    }

    const int tile = (int)(task % (unsigned)totalTiles);
    const int r = (tile / tilesN) * TM;
    const int c = (tile % tilesN) * TN;
    tile_gemm(A, B, C, K, N, r, c, N, r, c);
    __syncthreads();
  }
}

// Host<->globaltimer calibration probe: records the device wall clock the
// moment a host-written flag becomes visible.
__global__ void spin_flag(volatile int* flag, long long* out) {
  if (threadIdx.x == 0) {
    while (*flag == 0) {
    }
    *out = globaltimer_ns();
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

static void spin_for_ns(long long ns) {
  const long long deadline = now_ns() + ns;
  while (now_ns() < deadline) {
  }
}

static double percentile(std::vector<double> v, double q) {
  std::sort(v.begin(), v.end());
  return v[std::min((size_t)(q * v.size()), v.size() - 1)];
}

int main(int argc, char** argv) {
  const std::string mode = parse_str(argc, argv, "--mode", "latency");
  const std::string setBy = parse_str(argc, argv, "--set", "dev");
  const int K = (int)parse_arg(argc, argv, "--k", 64);
  const int tilesM = (int)parse_arg(argc, argv, "--tiles-m", 48);
  const int tilesN = (int)parse_arg(argc, argv, "--tiles-n", 48);
  const int events = (int)parse_arg(argc, argv, "--events", 10000);
  const unsigned tasks = (unsigned)parse_arg(argc, argv, "--tasks", 200000);
  const int reps = (int)parse_arg(argc, argv, "--reps", 20);
  const double gapMinUs = parse_arg(argc, argv, "--gap-min-us", 150);
  const double gapMaxUs = parse_arg(argc, argv, "--gap-max-us", 400);
  int blocks = (int)parse_arg(argc, argv, "--blocks", 0);
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
      &maxPerSM, persistent_yield<true>, THREADS, 0));
  if (blocks <= 0) blocks = sms * maxPerSM;

  const int M = tilesM * TM;
  const int N = tilesN * TN;
  const int totalTiles = tilesM * tilesN;

  printf("GPU: %s | %d SMs | %d blocks/SM -> sat=%d | mode=%s set=%s | "
         "K=%d blocks=%d\n",
         prop.name, sms, maxPerSM, sms * maxPerSM, mode.c_str(),
         setBy.c_str(), K, blocks);

  srand(21);
  std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
  for (auto& v : hA) v = (float)(rand() / (double)RAND_MAX * 2.0 - 1.0);
  for (auto& v : hB) v = (float)(rand() / (double)RAND_MAX * 2.0 - 1.0);

  float *dA, *dB, *dC, *dCu;
  unsigned int* dNext;
  int* dFlag;
  long long *dSched, *dSetGT, *dObsGT, *dUGT;
  int* dClaim;
  CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, (size_t)M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dCu, (size_t)TM * TN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dNext, sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc(&dFlag, sizeof(int)));
  const int evAlloc = std::max(events, 1);
  CUDA_CHECK(cudaMalloc(&dSched, (size_t)evAlloc * sizeof(long long)));
  CUDA_CHECK(cudaMalloc(&dSetGT, (size_t)evAlloc * sizeof(long long)));
  CUDA_CHECK(cudaMalloc(&dObsGT, (size_t)evAlloc * blocks * sizeof(long long)));
  CUDA_CHECK(cudaMalloc(&dUGT, (size_t)evAlloc * 2 * sizeof(long long)));
  CUDA_CHECK(cudaMalloc(&dClaim, (size_t)evAlloc * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  cudaStream_t kStream, cStream;  // kernel stream, control stream
  CUDA_CHECK(cudaStreamCreateWithFlags(&kStream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&cStream, cudaStreamNonBlocking));

  auto zero_all = [&]() {
    CUDA_CHECK(cudaMemset(dNext, 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(dFlag, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(dSetGT, 0, (size_t)evAlloc * sizeof(long long)));
    CUDA_CHECK(
        cudaMemset(dObsGT, 0, (size_t)evAlloc * blocks * sizeof(long long)));
    CUDA_CHECK(cudaMemset(dUGT, 0, (size_t)evAlloc * 2 * sizeof(long long)));
    CUDA_CHECK(cudaMemset(dClaim, 0, (size_t)evAlloc * sizeof(int)));
  };

  auto launch = [&](bool poll, unsigned nTasks, const long long* sched,
                    int nEvents) {
    if (poll)
      persistent_yield<true><<<blocks, THREADS, 0, kStream>>>(
          dA, dB, dC, dCu, K, N, tilesN, totalTiles, nTasks, dNext, dFlag,
          sched, nEvents, dSetGT, dObsGT, dClaim, dUGT);
    else
      persistent_yield<false><<<blocks, THREADS, 0, kStream>>>(
          dA, dB, dC, dCu, K, N, tilesN, totalTiles, nTasks, dNext, dFlag,
          nullptr, 0, dSetGT, dObsGT, dClaim, dUGT);
  };

  const int stopVal = -1;

  // ------------------------------------------------------------ overhead
  if (mode == "overhead") {
    // ABAB interleave so thermal / clock drift cancels. Flag never set.
    std::vector<double> wallPoll, wallNoPoll;
    for (int r = 0; r < 2 * reps; ++r) {
      const bool poll = (r % 2 == 0);
      CUDA_CHECK(cudaMemset(dNext, 0, sizeof(unsigned int)));
      CUDA_CHECK(cudaMemset(dFlag, 0, sizeof(int)));
      // Null-stream memsets do NOT order against a non-blocking stream, and
      // a saturated persistent kernel blocks the fill kernel entirely (the
      // all-NaN verify failure of 2026-08-04). Hard sync before launch.
      CUDA_CHECK(cudaDeviceSynchronize());
      const long long t0 = now_ns();
      launch(poll, tasks, nullptr, 0);
      CUDA_CHECK(cudaStreamSynchronize(kStream));
      const double ms = (now_ns() - t0) / 1e6;
      CUDA_CHECK(cudaGetLastError());
      (poll ? wallPoll : wallNoPoll).push_back(ms);
    }
    const double p = percentile(wallPoll, 0.5), n = percentile(wallNoPoll, 0.5);
    printf("drain of %u tasks, median of %d reps each (ABAB):\n", tasks, reps);
    printf("  poll ON : %.3f ms\n  poll OFF: %.3f ms\n", p, n);
    printf("  polling overhead: %.2f%%  (N1 cost threshold: 10%%)\n",
           (p / n - 1.0) * 100.0);
    if (csvPath[0]) {
      FILE* csv = fopen(csvPath, "w");
      if (csv) {
        fprintf(csv, "rep,poll,wall_ms\n");
        for (int r = 0; r < reps; ++r)
          fprintf(csv, "%d,1,%.4f\n%d,0,%.4f\n", r, wallPoll[r], r,
                  wallNoPoll[r]);
        fclose(csv);
        printf("raw: %s\n", csvPath);
      }
    }
    return 0;
  }

  // ------------------------------------------------------------ verify
  if (mode == "verify") {
    // Bounded run through the POLL code path with a handful of device-set
    // events, then float64 check of C and the last claimed urgent tile.
    const int vEvents = std::min(events, 5);
    std::vector<long long> gaps(evAlloc, 0);
    long long acc = 0;
    for (int e = 0; e < vEvents; ++e) gaps[e] = (acc += 300000);  // 300 us
    CUDA_CHECK(cudaMemcpy(dSched, gaps.data(),
                          (size_t)evAlloc * sizeof(long long),
                          cudaMemcpyHostToDevice));
    zero_all();
    CUDA_CHECK(cudaMemset(dC, 0xFF, (size_t)M * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(dCu, 0xFF, (size_t)TM * TN * sizeof(float)));
    CUDA_CHECK(cudaDeviceSynchronize());  // fills must land before launch
    launch(true, tasks, dSched, vEvents);
    CUDA_CHECK(cudaStreamSynchronize(kStream));
    CUDA_CHECK(cudaGetLastError());

    std::vector<float> hC((size_t)M * N), hCu((size_t)TM * TN);
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hCu.data(), dCu, hCu.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    std::vector<int> hClaim(evAlloc);
    CUDA_CHECK(cudaMemcpy(hClaim.data(), dClaim, evAlloc * sizeof(int),
                          cudaMemcpyDeviceToHost));

    const int coveredTiles = std::min((int)tasks, totalTiles);
    int bad = 0, nans = 0;
    double maxRel = 0.0;
    const double rtol = 1e-4, atol = 1e-4;
    int claimedEvents = 0;
    for (int e = 0; e < vEvents; ++e) claimedEvents += hClaim[e] != 0;
    for (int tile = 0; tile < coveredTiles; ++tile) {
      const int r0 = (tile / tilesN) * TM, c0 = (tile % tilesN) * TN;
      for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) {
          const int row = r0 + i, col = c0 + j;
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
          if (tile == 0 && claimedEvents > 0) check(hCu[(size_t)i * TN + j]);
        }
    }
    if (nans || bad) {
      fprintf(stderr, "FAIL: %d NaN, %d beyond tolerance, max rel %.3e\n",
              nans, bad, maxRel);
      return 1;
    }
    printf("PASS: %d tiles + urgent tile through the yield path (%d/%d "
           "events claimed), max rel err %.3e\n",
           coveredTiles, claimedEvents, vEvents, maxRel);
    return 0;
  }

  // ------------------------------------------------------------ latency
  if (mode != "latency") {
    fprintf(stderr, "unknown --mode %s\n", mode.c_str());
    return 1;
  }

  // Host<->device clock calibration (host mode). Idle-GPU measurement; the
  // dev-set mode exists to bound what load does to this number.
  long long offP50 = 0, offMin = 0, offP90 = 0;
  if (setBy == "host") {
    long long* dOut;
    CUDA_CHECK(cudaMalloc(&dOut, sizeof(long long)));
    std::vector<double> offs;
    const int calReps = 300;
    for (int r = 0; r < calReps; ++r) {
      CUDA_CHECK(cudaMemset(dFlag, 0, sizeof(int)));
      spin_flag<<<1, 32, 0, kStream>>>(dFlag, dOut);
      spin_for_ns(200000);  // let the spin kernel actually start
      const long long h0 = now_ns();
      const int one = 1;
      CUDA_CHECK(cudaMemcpyAsync(dFlag, &one, sizeof(int),
                                 cudaMemcpyHostToDevice, cStream));
      CUDA_CHECK(cudaStreamSynchronize(kStream));
      long long gObs;
      CUDA_CHECK(
          cudaMemcpy(&gObs, dOut, sizeof(long long), cudaMemcpyDeviceToHost));
      offs.push_back((double)(gObs - h0));
    }
    offMin = (long long)percentile(offs, 0.0);
    offP50 = (long long)percentile(offs, 0.5);
    offP90 = (long long)percentile(offs, 0.9);
    printf("host->device flag visibility (globaltimer - host ns, %d reps): "
           "min %.1f us | p50 %.1f | p90 %.1f\n",
           calReps, offMin / 1e3, offP50 / 1e3, offP90 / 1e3);
    CUDA_CHECK(cudaFree(dOut));
  }

  // Event schedule: cumulative gaps, U(gapMin, gapMax), fixed seed.
  std::mt19937 rng(21);
  std::uniform_real_distribution<double> gapDist(gapMinUs * 1000.0,
                                                 gapMaxUs * 1000.0);
  std::vector<long long> gaps(evAlloc);
  long long acc = 0;
  for (int e = 0; e < events; ++e) gaps[e] = (acc += (long long)gapDist(rng));
  CUDA_CHECK(cudaMemcpy(dSched, gaps.data(),
                        (size_t)evAlloc * sizeof(long long),
                        cudaMemcpyHostToDevice));
  zero_all();

  std::vector<long long> hSet(events, 0);  // host-set timestamps (host ns)
  const bool devSet = (setBy == "dev");

  CUDA_CHECK(cudaDeviceSynchronize());  // zeroing must land before launch
  launch(true, 0xFFFFFFF0u, devSet ? dSched : nullptr, devSet ? events : 0);

  if (devSet) {
    // Setter runs on-device; host just waits out the schedule.
    spin_for_ns(acc + 100 * 1000 * 1000LL);
  } else {
    spin_for_ns(50 * 1000 * 1000LL);  // settle into steady state
    for (int e = 0; e < events; ++e) {
      spin_for_ns(e == 0 ? gaps[0] : gaps[e] - gaps[e - 1]);
      const int val = e + 1;
      hSet[e] = now_ns();
      CUDA_CHECK(cudaMemcpyAsync(dFlag, &val, sizeof(int),
                                 cudaMemcpyHostToDevice, cStream));
    }
    CUDA_CHECK(cudaStreamSynchronize(cStream));
    spin_for_ns(5 * 1000 * 1000LL);
  }
  CUDA_CHECK(cudaMemcpyAsync(dFlag, &stopVal, sizeof(int),
                             cudaMemcpyHostToDevice, cStream));
  CUDA_CHECK(cudaStreamSynchronize(kStream));
  CUDA_CHECK(cudaGetLastError());

  std::vector<long long> hSetGT(evAlloc), hUGT((size_t)evAlloc * 2);
  std::vector<long long> hObs((size_t)evAlloc * blocks);
  std::vector<int> hClaim(evAlloc);
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

  std::vector<double> lat, spread, e2e, firstDelay;
  int anomalies = 0, unset = 0;
  FILE* csv = csvPath[0] ? fopen(csvPath, "w") : nullptr;
  if (csv)
    fprintf(csv,
            "event,set_ns,first_obs_ns,last_obs_ns,lat_us,spread_us,"
            "claimer,e2e_us,complete\n");
  for (int e = 0; e < events; ++e) {
    long long setNs;
    if (devSet) {
      setNs = hSetGT[e];
      if (setNs == 0) { ++unset; continue; }  // schedule outran the run
    } else {
      setNs = hSet[e] + offP50;  // host clock mapped onto globaltimer
    }
    long long lo = 0, hi = 0;
    int seen = 0;
    for (int b = 0; b < blocks; ++b) {
      const long long o = hObs[(size_t)e * blocks + b];
      if (o == 0) continue;
      ++seen;
      lo = (lo == 0 || o < lo) ? o : lo;
      hi = std::max(hi, o);
    }
    const bool complete = (seen == blocks);
    if (!complete) { ++anomalies; }
    const double latUs = (hi - setNs) / 1e3;
    const double spreadUs = (hi - lo) / 1e3;
    const double e2eUs =
        hClaim[e] ? (hUGT[2 * (size_t)e + 1] - setNs) / 1e3 : -1.0;
    if (complete) {
      lat.push_back(latUs);
      spread.push_back(spreadUs);
      firstDelay.push_back((lo - setNs) / 1e3);
      if (e2eUs >= 0) e2e.push_back(e2eUs);
    }
    if (csv)
      fprintf(csv, "%d,%lld,%lld,%lld,%.2f,%.2f,%d,%.2f,%d\n", e, setNs, lo,
              hi, latUs, spreadUs, hClaim[e] - 1, e2eUs, complete ? 1 : 0);
  }
  if (csv) fclose(csv);

  printf("\n--- latency (%s-set): %zu complete events (%d anomalies, %d "
         "unset) ---\n",
         setBy.c_str(), lat.size(), anomalies, unset);
  if (!lat.empty()) {
    printf("preemption latency set->last (us): p50 %.2f | p90 %.2f | p99 "
           "%.2f | max %.2f\n",
           percentile(lat, 0.5), percentile(lat, 0.9), percentile(lat, 0.99),
           *std::max_element(lat.begin(), lat.end()));
    printf("first observer delay (us):        p50 %.2f | p99 %.2f\n",
           percentile(firstDelay, 0.5), percentile(firstDelay, 0.99));
    printf("observation spread first->last (us): p50 %.2f | p99 %.2f\n",
           percentile(spread, 0.5), percentile(spread, 0.99));
    if (!e2e.empty())
      printf("urgent e2e set->done (us): p50 %.2f | p99 %.2f  [Phase 2 sat "
             "baseline: 957 p50 / 1436 p99]\n",
             percentile(e2e, 0.5), percentile(e2e, 0.99));
  }
  if (csvPath[0]) printf("raw: %s\n", csvPath);
  return 0;
}
