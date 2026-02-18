#include "bench.h"
#include <thread>
#include <cassert>

constexpr uint64_t SIZE = 1e7;

namespace bench {
    void bench_dot() {
        Vec a(SIZE, 2.0f);
        Vec b(SIZE, 2.0f);

        double t_simple = time_function([&] {
            kernels::dot_simple(a, b);
        });


        uint32_t num_threads = std::thread::hardware_concurrency();
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