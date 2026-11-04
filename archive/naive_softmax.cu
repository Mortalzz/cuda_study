// 01_reduction/naive_reduction.cu
//
// NHWC softmax along the channel dimension.
// One thread normalizes one (batch, m, n) position, i.e. `c` contiguous floats.
//
// Fixes vs. the previous version:
//   1. linear buffers are sized in 64-bit (int overflowed to 0 -> null pointers)
//   2. every allocation / memcpy is wrapped in CUDA_CHECK
//   3. the element offset accounts for blockIdx.x (previously blockIdx.x was lost,
//      so all blocks raced on the same 32 rows)
//   4. a real bounds check keeps the tail block inside the buffer
//   5. max starts at -FLT_MAX and the sum is taken after subtracting max
//      (previously exp(1000) -> inf -> sum = inf -> inv = 0 -> all zeros)
//   6. the reference comparison uses a tolerance (__expf != expf bitwise)

#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <float.h>
#include <iostream>

#include "../common/cuda_check.cuh"
#include "../common/reference.h"
#include "../common/timer.cuh"

#define BLOCK_ROWS 256

// ---------------------------------------------------------------------------
// Kernel: one thread per row of `c` contiguous channels.
// ---------------------------------------------------------------------------
__global__ void naive_softmax_NHWC(const float* __restrict__ a,
                                   float* __restrict__ b,
                                   int rows,
                                   int c) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;                                // tail guard

    const size_t off = static_cast<size_t>(row) * static_cast<size_t>(c);  // 64-bit

    float max = -FLT_MAX;
    for (int i = 0; i < c; ++i) max = fmaxf(max, a[off + i]);

    float sum = 0.0f;
    for (int i = 0; i < c; ++i) sum += __expf(a[off + i] - max);

    const float inv = 1.0f / sum;
    for (int i = 0; i < c; ++i) b[off + i] = __expf(a[off + i] - max) * inv;
}

// ---------------------------------------------------------------------------
// Two checks instead of `a[i] != b[i]`:
//   1. element-wise |gpu - cpu| <= atol + rtol*|cpu|
//   2. every softmax row must sum to 1
// (1) alone is weak because softmax values span many orders of magnitude, and
// exact == never holds between __expf and expf.
// ---------------------------------------------------------------------------
struct CheckResult {
    bool   ok{true};
    size_t first_bad{0};
    double max_rel_err{0.0};   // largest relative error over ref > 1e-6
    double max_row_dev{0.0};   // largest |rowsum - 1|
};

static CheckResult check_softmax(const float* got, const float* ref,
                                 int rows, int c, float rtol, float atol) {
    CheckResult res;
    res.first_bad = static_cast<size_t>(rows) * c;

    for (size_t i = 0, n = static_cast<size_t>(rows) * c; i < n; ++i) {
        const double g = got[i];
        const double r = ref[i];
        const double diff = std::fabs(g - r);
        if (res.ok && !(diff <= atol + rtol * std::fabs(r))) {
            res.ok = false;
            res.first_bad = i;
        }
        if (std::fabs(r) > 1e-6) {
            const double rel = diff / std::fabs(r);
            if (rel > res.max_rel_err) res.max_rel_err = rel;
        }
    }

    for (int row = 0; row < rows; ++row) {
        double sum = 0.0;
        const size_t off = static_cast<size_t>(row) * c;
        for (int i = 0; i < c; ++i) sum += got[off + i];
        const double dev = std::fabs(sum - 1.0);
        if (dev > res.max_row_dev) res.max_row_dev = dev;
    }
    if (!(res.max_row_dev <= 1e-4)) res.ok = false;
    return res;
}

// ---------------------------------------------------------------------------
// Host driver
// ---------------------------------------------------------------------------
void naive_reduction_channel(int batch, int m, int n, int c) {
    // ---- 64-bit size math, then prove it fits the int-based reference API ----
    const long long rows_ll = static_cast<long long>(batch) * m * n;
    const long long sz_ll   = rows_ll * c;

    if (rows_ll <= 0 || c <= 0) {
        std::fprintf(stderr, "invalid problem size\n");
        std::exit(EXIT_FAILURE);
    }
    if (sz_ll > INT_MAX) {
        std::fprintf(stderr,
                     "problem too large for this harness: %lld floats (%.1f GiB).\n"
                     "common/reference.h indexes with `int`, so keep the total under %d elements.\n",
                     sz_ll, static_cast<double>(sz_ll) * 4.0 / (1024.0 * 1024.0 * 1024.0), INT_MAX);
        std::exit(EXIT_FAILURE);
    }

    const int rows  = static_cast<int>(rows_ll);
    const int sz    = static_cast<int>(sz_ll);
    const size_t bytes = static_cast<size_t>(sz) * sizeof(float);

    std::printf("rows = %d, c = %d, elements = %d (%.1f MiB per buffer)\n",
                rows, c, sz, static_cast<double>(bytes) / (1024.0 * 1024.0));

    // ---- host (pinned) + device buffers, every call checked in 64-bit ----
    float* a    = nullptr;   // input  (host)
    float* b    = nullptr;   // output (host)
    float* comp = nullptr;   // output (CPU reference)
    float* dev_a = nullptr;
    float* dev_b = nullptr;

    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&a),    bytes));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&b),    bytes));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&comp), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_a), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_b), bytes));

    Init(a, sz);
    Init(b, sz, false);
    Init(comp, sz, false);

    CUDA_CHECK(cudaMemcpy(dev_a, a, bytes, cudaMemcpyHostToDevice));

    const dim3 block(BLOCK_ROWS, 1, 1);
    const dim3 grid(static_cast<unsigned>((rows + BLOCK_ROWS - 1) / BLOCK_ROWS), 1, 1);

    CudaTimer timer;
    timer.start();
    naive_softmax_NHWC<<<grid, block>>>(dev_a, dev_b, rows, c);
    CUDA_CHECK(cudaGetLastError());   // launch-config errors show up here, not in the timer
    timer.stop();
    const double ms = timer.elapsed_ms();

    CUDA_CHECK(cudaMemcpy(b, dev_b, bytes, cudaMemcpyDeviceToHost));

    softmax_cpu_NHWC(comp, a, batch, m, n, c);

    const CheckResult res = check_softmax(b, comp, rows, c, 1e-4f, 1e-7f);

    if (res.ok) {
        std::cout << "Functional Naive_Softmax_NHWC Testing PASS!!!" << std::endl;
        std::printf("max relative error = %.3e, max |rowsum-1| = %.3e\n",
                    res.max_rel_err, res.max_row_dev);
    } else {
        std::cout << "Functional Naive_Softmax_NHWC Testing WRONG!!!" << std::endl;
        std::printf("first mismatch at %zu (row %zu, ch %zu): got %g, ref %g\n",
                    res.first_bad, res.first_bad / c, res.first_bad % c,
                    b[res.first_bad], comp[res.first_bad]);
        std::printf("max relative error = %.3e, max |rowsum-1| = %.3e\n",
                    res.max_rel_err, res.max_row_dev);
    }

    const double gbs = 2.0 * static_cast<double>(bytes) / (ms * 1e6);
    std::printf("Final Speed: %f ms\n%.3f GB/s\n", ms, gbs);

    CUDA_CHECK(cudaFree(dev_a));
    CUDA_CHECK(cudaFree(dev_b));
    CUDA_CHECK(cudaFreeHost(a));
    CUDA_CHECK(cudaFreeHost(b));
    CUDA_CHECK(cudaFreeHost(comp));
}

int main(int argc, char** argv) {
    // Default: 16 * 128 * 128 rows x 256 channels = 2^26 floats = 256 MiB per buffer.
    // (The old 256x256x256x256 call is 2^32 floats = 16 GiB per buffer, which
    //  cannot fit in a 16 GB card and does not even fit in an `int`.)
    //
    // Override on the command line: naive_reduction <batch> <m> <n> <c>
    int batch = 16, m = 128, n = 128, c = 256;
    if (argc == 5) {
        batch = std::atoi(argv[1]);
        m     = std::atoi(argv[2]);
        n     = std::atoi(argv[3]);
        c     = std::atoi(argv[4]);
    } else if (argc != 1) {
        std::fprintf(stderr, "usage: %s [batch m n c]\n", argv[0]);
        return EXIT_FAILURE;
    }

    naive_reduction_channel(batch, m, n, c);
    return 0;
}
