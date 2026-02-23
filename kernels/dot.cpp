#include <cstdint>
#include <immintrin.h>
#include <pmmintrin.h>
#include <thread>
#include <vector>
#include "../lib/vec.h"
#include "dot.h"

struct alignas(64) PaddedFloat {
    float value;
};

namespace {
    void dot_simple_kernel(const Vec& a, const Vec& b, PaddedFloat* results, uint32_t thread_id, uint32_t start, uint32_t end) {
        float localSum = 0.0f;
        for(uint32_t i = start; i < end; i++) {
            localSum += a[i] * b[i];
        }
        results[thread_id].value = localSum;
    }

    void dot_simd_kernel(const Vec& a, const Vec& b, PaddedFloat* results, uint32_t thread_id, uint32_t start, uint32_t end) {
        __m128 sum = _mm_setzero_ps();
        size_t i = start;
        for (; i + 3 < end; i += 4) {
            __m128 prod = _mm_mul_ps(_mm_load_ps(a.data() + i), _mm_load_ps(b.data() + i));
            sum = _mm_add_ps(sum, prod);
        }

        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);

        float result = _mm_cvtss_f32(sum);

        for (; i < end; i++) {
            result += a[i] * b[i];
        }

        results[thread_id].value = result;
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
        __m128 sum = _mm_setzero_ps();
        size_t i = 0;
        for (; i + 3 < a.size(); i += 4) {
            __m128 prod = _mm_mul_ps(_mm_load_ps(a.data() + i), _mm_load_ps(b.data() + i));
            sum = _mm_add_ps(sum, prod);
        }

        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);

        float result = _mm_cvtss_f32(sum);
        
        for (size_t i = (a.size() / 4) * 4; i < a.size(); i++) {
            result += a[i] * b[i];
        }
        
        return result;
    }

    float dot_parallel(const Vec& a, const Vec& b, uint32_t num_threads) {
        std::vector<std::thread> threads(num_threads);
        std::vector<PaddedFloat> results(num_threads);

        for (uint32_t i = 0; i < num_threads; i++) {
            results[i].value = 0.0f;
        }

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
            result += results[i].value;
        }

        return result;
    }
    
    float dot_simd_parallel(const Vec& a, const Vec& b, uint32_t num_threads) {
        if (a.size() < num_threads * 1024)
            num_threads = 1;

        std::vector<std::thread> threads(num_threads);
        std::vector<PaddedFloat> results(num_threads);

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
            result += results[i].value;
        }

        return result;
    }
}
