#include <cstdint>
#include <thread>
#include <vector>
#include "../lib/vec_naive.h"
#include "dot.h"

namespace {
    void dot_kernel(const naive::Vec& a, const naive::Vec& b, float* results, uint32_t thread_id, uint32_t start, uint32_t end) {
        float localSum = 0.0f;
        for(uint32_t i = start; i < end; i++) {
            localSum += a[i] * b[i];
        }
        results[thread_id] = localSum;
    }
}


namespace kernels {
    float dot_parallel(const naive::Vec& a, const naive::Vec& b, uint32_t num_threads) {
        std::vector<std::thread> threads(num_threads);
        std::vector<float> results(num_threads);

        for (uint32_t i = 0; i < num_threads; i++) {
            uint32_t start = i * a.size() / num_threads;
            uint32_t end   = (i + 1) * a.size() / num_threads;

            threads[i] = std::thread(
                dot_kernel,
                std::cref(a),
                std::cref(b),
                results.data(),
                i,
                start,
                end
            );
        }

        for (auto& t : threads)
            t.join();

        float result = 0.0f;
        for (uint32_t i = 0; i < num_threads; i++) {
            result += results[i];
        }

        return result;
    }
}