#pragma once
#include <algorithm>
#include <cstdio>
#include <vector>

#include "timer.cuh"

struct BenchResult {
    double mean_ms;
    double median_ms;
    double min_ms;
    double max_ms;
    double gbps;
};

template <typename Launch>
BenchResult benchmark(const char* name,
                      Launch launch,
                      int warmup,
                      int iters,
                      double bytes)
{
    // 1. 预热：让 GPU 进入稳定频率/功耗状态
    for (int i = 0; i < warmup; ++i) {
        launch();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 2. 每次迭代单独计时
    CudaTimer timer;
    std::vector<double> samples;
    samples.reserve(iters);

    for (int i = 0; i < iters; ++i) {
        timer.start();
        launch();
        timer.stop();

        CUDA_CHECK(cudaGetLastError());
        samples.push_back(timer.elapsed_ms());
    }

    // 3. 统计
    std::sort(samples.begin(), samples.end());

    double sum = 0.0;
    for (double t : samples) sum += t;

    double mean   = sum / samples.size();
    double median = samples[samples.size() / 2];
    double min_ms = samples.front();
    double max_ms = samples.back();

    // bytes / (ms * 1e-3) / 1e9 = GB/s
    double gbps = bytes / (median * 1e-3) / 1e9;

    std::printf(
        "%-18s mean=%8.4f  median=%8.4f  min=%8.4f  max=%8.4f  %8.2f GB/s\n",
        name, mean, median, min_ms, max_ms, gbps);

    return {mean, median, min_ms, max_ms, gbps};
}
