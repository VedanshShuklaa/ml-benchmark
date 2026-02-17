#include <cstdint>
#include <immintrin.h>
#include <thread>
#include <vector>
#include "../lib/vec.h"
#include "dot.h"

namespace {
    void dot_simple_kernel(const Vec& a, const Vec& b, float* results, uint32_t thread_id, uint32_t start, uint32_t end) {
        float localSum = 0.0f;
        for(uint32_t i = start; i < end; i++) {
            localSum += a[i] * b[i];
        }
        results[thread_id] = localSum;
    }

    void dot_simd_kernel(const Vec& a, const Vec& b, float* results, uint32_t thread_id, uint32_t start, uint32_t end) {
        __m128 v1 = _mm_load_ps(a.data() + start);
        __m128 v2 = _mm_load_ps(b.data() + start);

        __m128 sum = _mm_setzero_ps();
        for (size_t i = start; i < end; i += 4) {
            __m128 prod = _mm_mul_ps(v1, v2);
            sum = _mm_add_ps(sum, prod);
        }
        results[thread_id] = _mm_cvtss_f32(sum);
    }
}

namespace kernels {
    float dot_simple(const Vec& a, const Vec& b) {
        float result = 0.0f;
        for(uint32_t i = 0; i < a.size(); i++) {
            result += a[i] * b[i];
        }
        return result;
    }

    float dot_simd(const Vec& a, const Vec& b) {
        __m128 v1 = _mm_load_ps(a.data());
        __m128 v2 = _mm_load_ps(b.data());

        __m128 sum = _mm_setzero_ps();
        for (size_t i = 0; i < a.size(); i += 4) {
            __m128 prod = _mm_mul_ps(v1, v2);
            sum = _mm_add_ps(sum, prod);
        }
        return _mm_cvtss_f32(sum);
    }

    float dot_parallel(const Vec& a, const Vec& b, uint32_t num_threads) {
        std::vector<std::thread> threads(num_threads);
        std::vector<float> results(num_threads);

        for (uint32_t i = 0; i < num_threads; i++) {
            uint32_t start = i * a.size() / num_threads;
            uint32_t end   = (i + 1) * a.size() / num_threads;

            threads[i] = std::thread(
                dot_simple_kernel,
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
    
    float dot_simd_parallel(const Vec& a, const Vec& b, uint32_t num_threads) {
        if (a.size() < num_threads * 1024)
            num_threads = 1;

        std::vector<std::thread> threads(num_threads);
        std::vector<float> results(num_threads);

        for (uint32_t i = 0; i < num_threads; i++) {
            uint32_t start = i * a.size() / num_threads;
            uint32_t end   = (i + 1) * a.size() / num_threads;

            threads[i] = std::thread(
                dot_simd_kernel,
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
