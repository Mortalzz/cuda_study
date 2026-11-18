#pragma once
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cstddef>
#include <cstdint>

// Independent example for SM 80+. Row-major C = A * B; A/B/C do not alias.
// No changes to 04_SGEMM/sgemm.cu or the project build configuration.
namespace sgemm_pipeline {
constexpr int BM = 128, BN = 128, BK = 32, RT = 4;
constexpr int THREADS = 1024;
constexpr int SLOT_FLOATS = BM * BK + BK * BN;
constexpr int SHARED_BYTES = 2 * SLOT_FLOATS * sizeof(float); // 64 KiB

// Each thread copies distinct elements. All threads commit/wait outside here.
// Aligned rows use 16-byte copies; other rows use 4-byte copies.
template<int ROWS, int COLS>
__device__ __forceinline__ void copy_tile_async(
    float (*dst)[COLS], const float* src,
    int row0, int col0, int rows, int cols, int tid)
{
    if ((cols % 4 == 0) &&
        ((reinterpret_cast<std::uintptr_t>(src) & 15u) == 0)) {
        for (int v = tid; v < ROWS * COLS / 4; v += THREADS) {
            int r = v / (COLS / 4), c = (v % (COLS / 4)) * 4;
            int gr = row0 + r, gc = col0 + c;
            bool valid = gr < rows && gc + 3 < cols;
            // Even an entirely masked copy uses a valid source pointer.
            const float* p = valid ? src + std::size_t(gr) * cols + gc : src;
            __pipeline_memcpy_async(&dst[r][c], p, 16, valid ? 0 : 16);
        }
    } else {
        for (int v = tid; v < ROWS * COLS; v += THREADS) {
            int r = v / COLS, c = v % COLS;
            int gr = row0 + r, gc = col0 + c;
            bool valid = gr < rows && gc < cols;
            const float* p = valid ? src + std::size_t(gr) * cols + gc : src;
            __pipeline_memcpy_async(&dst[r][c], p, 4, valid ? 0 : 4);
        }
    }
}

__device__ __forceinline__ void read_fragment(
    const float (*a)[BK], const float (*b)[BN],
    int kk, int row, int col, float (&ar)[RT], float (&br)[RT])
{
    #pragma unroll
    for (int i = 0; i < RT; ++i) ar[i] = a[row + i][kk];
    #pragma unroll
    for (int j = 0; j < RT; ++j) br[j] = b[kk][col + j];
}

__device__ __forceinline__ void outer_product(
    const float (&ar)[RT], const float (&br)[RT], float (&acc)[RT][RT])
{
    #pragma unroll
    for (int i = 0; i < RT; ++i) {
        #pragma unroll
        for (int j = 0; j < RT; ++j)
            acc[i][j] = fmaf(ar[i], br[j], acc[i][j]);
    }
}

// Method 2 uses the original style of shared-memory consumption.
__device__ __forceinline__ void compute_plain(
    const float (*a)[BK], const float (*b)[BN],
    int row, int col, float (&acc)[RT][RT])
{
    #pragma unroll 2
    for (int kk = 0; kk < BK; ++kk) {
        float ar[RT], br[RT];
        read_fragment(a, b, kk, row, col, ar, br);
        outer_product(ar, br, acc);
    }
}

// Method 3: two register fragments, one consumed while the other is prefetched.
// Can also replace ONLY the kk loop in the original kernel; keep its barriers.
__device__ __forceinline__ void compute_prefetched(
    const float (*a)[BK], const float (*b)[BN],
    int row, int col, float (&acc)[RT][RT])
{
    float ar[2][RT], br[2][RT];
    read_fragment(a, b, 0, row, col, ar[0], br[0]);

    #pragma unroll
    for (int kk = 0; kk < BK - 1; ++kk) {
        // Unrolling makes both register-buffer indices compile-time constants.
        int cur = kk & 1, next = cur ^ 1;
        read_fragment(a, b, kk + 1, row, col, ar[next], br[next]);
        outer_product(ar[cur], br[cur], acc);
    }
    outer_product(ar[(BK - 1) & 1], br[(BK - 1) & 1], acc);
}

// false: method 2. true: method 2 + method 3.
template<bool RegisterPrefetch>
__global__ __launch_bounds__(THREADS) void kernel(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ c, int m, int n, int k)
{
    extern __shared__ __align__(16) float smem[];
    auto sa = reinterpret_cast<float (*)[BM][BK]>(smem);
    auto sb = reinterpret_cast<float (*)[BK][BN]>(smem + 2 * BM * BK);
    int tid = threadIdx.y * 32 + threadIdx.x;
    int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    int row = threadIdx.y * RT, col = threadIdx.x * RT;
    float acc[RT][RT] = {};
    int tiles = k / BK + (k % BK != 0);

    if (tiles > 0) {
        // Prologue: fill the first slot before consuming it.
        copy_tile_async<BM, BK>(sa[0], a, row0, 0, m, k, tid);
        copy_tile_async<BK, BN>(sb[0], b, 0, col0, k, n, tid);
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();

        for (int tile = 0; tile < tiles; ++tile) {
            int cur = tile & 1, next = cur ^ 1;
            bool has_next = tile + 1 < tiles;
            if (has_next) {
                // Submit next tile before computing current tile.
                int k0 = (tile + 1) * BK;
                copy_tile_async<BM, BK>(sa[next], a, row0, k0, m, k, tid);
                copy_tile_async<BK, BN>(sb[next], b, k0, col0, k, n, tid);
                __pipeline_commit();
            }

            if constexpr (RegisterPrefetch)
                compute_prefetched(sa[cur], sb[cur], row, col, acc);
            else
                compute_plain(sa[cur], sb[cur], row, col, acc);

            if (has_next) {
                __pipeline_wait_prior(0); // This thread's copies completed.
                __syncthreads();          // All next data ready; all cur readers done.
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < RT; ++i) {
        #pragma unroll
        for (int j = 0; j < RT; ++j) {
            int gr = row0 + row + i, gc = col0 + col + j;
            if (gr < m && gc < n) c[std::size_t(gr) * n + gc] = acc[i][j];
        }
    }
}

// Call configure once, outside the timed region. SM 80+ is required.
template<bool RegisterPrefetch>
inline cudaError_t configure()
{
    return cudaFuncSetAttribute(kernel<RegisterPrefetch>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_BYTES);
}

// M/N=0: no work. K=0: writes zeros. Positive extents require valid device buffers.
template<bool RegisterPrefetch>
inline cudaError_t launch(const float* a, const float* b, float* c,
                         int m, int n, int k, cudaStream_t stream = nullptr)
{
    if (m < 0 || n < 0 || k < 0) return cudaErrorInvalidValue;
    if (m == 0 || n == 0) return cudaSuccess;
    dim3 grid(n / BN + (n % BN != 0), m / BM + (m % BM != 0));
    kernel<RegisterPrefetch><<<grid, dim3(32, 32), SHARED_BYTES, stream>>>(
        a, b, c, m, n, k);
    return cudaGetLastError();
}
} // namespace sgemm_pipeline
