// Shared tile machinery for all phases: one TM x TN SGEMM tile computed
// cooperatively by THREADS threads, K as the duration knob (calibrated
// 2026-08-04 at 1200 MHz: K=32 -> 6.7 us, K=64 -> 10.3 us, K=128 -> 17.8 us).
#pragma once
#include <cuda_runtime.h>

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
