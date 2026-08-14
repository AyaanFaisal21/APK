// The synthetic tile — the one unit of work every binary in this repo
// schedules, preempts, and measures. One TM x TN SGEMM tile computed
// cooperatively by THREADS threads, K as the duration knob (calibrated by
// persistent.cu, 2026-08-04 at 1200 MHz: K=32 -> 6.7 us, K=64 -> 10.3 us,
// K=128 -> 17.8 us; solo-SM figures — 4-way co-residency stretches walls
// to a 57.6 us p99, which is what preemption latency actually waits on).
//
// Changing anything here invalidates that calibration and every number
// downstream of it. Recalibrate (scripts/calibrate.sh) before trusting a
// sweep run against a modified tile.
#pragma once
#include <cuda_runtime.h>
#include <cuda/atomic>

// Flag contract (post-review, 2026-08-08): the generation flag is a
// device-scope atomic that communicates ONLY the generation value.
// Nothing is published through it: claims go through atomicCAS, and
// setGT is read by the host only after stream synchronization. Loads
// and stores are therefore memory_order_relaxed by design, not by
// omission. CUDA C++ volatile carries no inter-thread synchronization
// guarantee, so the previous volatile protocol was unspecified even
// though it behaved identically on sm_86.
using flag_ref = cuda::atomic_ref<int, cuda::thread_scope_device>;

__device__ __forceinline__ int flag_load(int* f) {
  return flag_ref(*f).load(cuda::memory_order_relaxed);
}
__device__ __forceinline__ void flag_store(int* f, int v) {
  flag_ref(*f).store(v, cuda::memory_order_relaxed);
}

constexpr int TM = 64;
constexpr int TN = 64;
constexpr int KB = 32;
constexpr int THREADS = 256;

struct TileRecord {
  long long start;
  long long end;
  int smid;
  int block;
};

__device__ __forceinline__ unsigned smid_of() {
  unsigned id;
  asm("mov.u32 %0, %%smid;" : "=r"(id));
  return id;
}

// Nanosecond wall clock, coherent across SMs and across kernels — unlike
// clock64(), which is a per-SM cycle counter. Use this for anything that
// compares timestamps between two kernels (e.g. urgent start vs bg drain).
__device__ __forceinline__ long long globaltimer_ns() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// C tile at (dstRow, dstCol) [leading dim ldC] = A rows [srcRow, srcRow+TM)
// times B cols [srcCol, srcCol+TN). All THREADS threads must call this.
// +1 padding on the shared arrays keeps column reads off a single bank.
__device__ void tile_gemm(const float* __restrict__ A,
                          const float* __restrict__ B, float* __restrict__ C,
                          int K, int N, int srcRow, int srcCol, int ldC,
                          int dstRow, int dstCol) {
  __shared__ float As[TM][KB + 1];
  __shared__ float Bs[KB][TN + 1];

  const int tx = threadIdx.x % 16;
  const int ty = threadIdx.x / 16;

  float acc[4][4] = {};

  for (int k0 = 0; k0 < K; k0 += KB) {
    for (int i = threadIdx.x; i < TM * KB; i += THREADS) {
      const int r = i / KB, c = i % KB;
      As[r][c] = A[(long long)(srcRow + r) * K + (k0 + c)];
    }
    for (int i = threadIdx.x; i < KB * TN; i += THREADS) {
      const int r = i / TN, c = i % TN;
      Bs[r][c] = B[(long long)(k0 + r) * N + (srcCol + c)];
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < KB; ++kk) {
      float a[4], b[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) a[i] = As[ty * 4 + i][kk];
#pragma unroll
      for (int j = 0; j < 4; ++j) b[j] = Bs[kk][tx * 4 + j];
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const long long r = dstRow + ty * 4 + i;
#pragma unroll
    for (int j = 0; j < 4; ++j)
      C[r * ldC + (dstCol + tx * 4 + j)] = acc[i][j];
  }
}
