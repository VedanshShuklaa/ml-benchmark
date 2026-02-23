#include "bench.h"
#include <thread>
#include <random>
#include <cassert>
#include <cmath>

constexpr uint64_t SIZE = 1e7;
constexpr float EPSILON = 0.1;

namespace bench {
    inline bool nearly_equal(float a, float b) {
        return std::abs(a - b) < EPSILON;
    }

    inline void fill_random(Vec& v) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        for (uint64_t i = 0; i < v.size(); i++) {
            v[i] = dis(gen);
        }
    }

    void bench_dot() {
        Vec a(SIZE);
        Vec b(SIZE);
        fill_random(a);
        fill_random(b);

        uint32_t num_threads = std::thread::hardware_concurrency();

        float sum_simple = kernels::dot_simple(a, b);
        float sum_parallel = kernels::dot_parallel(a, b, num_threads);
        float sum_simd = kernels::dot_simd(a, b);
        float sum_simd_parallel = kernels::dot_simd_parallel(a, b, num_threads);

        assert(nearly_equal(sum_simple, sum_parallel));
        assert(nearly_equal(sum_simple, sum_simd));
        assert(nearly_equal(sum_simple, sum_simd_parallel));

        double t_simple = time_function([&] {
            kernels::dot_simple(a, b);
        });

        double t_parallel = time_function([&] {
            kernels::dot_parallel(a, b, num_threads);
        });

        double t_simd = time_function([&] {
            kernels::dot_simd(a, b);
        });
        
        double t_simd_parallel = time_function([&] {
            kernels::dot_simd_parallel(a, b, num_threads);
        });
        
        std::cout << "Simple dot product took " << t_simple * 1000000 << " microseconds" << std::endl;
        std::cout << "Parallel dot product took " << t_parallel * 1000000 << " microseconds" << std::endl;
        std::cout << "SIMD dot product took " << t_simd * 1000000 << " microseconds" << std::endl;
        std::cout << "SIMD parallel dot product took " << t_simd_parallel * 1000000 << " microseconds" << std::endl;
    }
}