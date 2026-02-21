#include "bench.h"
#include <thread>
#include <cassert>

constexpr uint64_t SIZE = 512;

namespace bench {
    void bench_gemm() {
        Mat a(SIZE, SIZE, 2.0f);
        Mat b(SIZE, SIZE, 2.0f);

        double t_simple = time_function([&] {
            kernels::gemm_simple(a, b);
        });


        uint32_t num_threads = std::thread::hardware_concurrency();
        double t_parallel = time_function([&] {
            kernels::gemm_parallel(a, b, num_threads);
        });

        double t_omp = time_function([&] {
            kernels::gemm_omp(a, b);
        });

        // double t_simd = time_function([&] {
        //     kernels::gemm_simd(a, b);
        // });
        
        // double t_simd_parallel = time_function([&] {
        //     kernels::gemm_simd_parallel(a, b, num_threads);
        // });
        
        std::cout << "Simple gemm product took " << t_simple * 1000000 << " microseconds" << std::endl;

        std::cout << "Parallel gemm product took " << t_parallel * 1000000 << " microseconds" << std::endl;

        std::cout << "OMP Parallel gemm product took " << t_omp * 1000000 << " microseconds" << std::endl;
        // std::cout << "SIMD gemm product took " << t_simd * 1000000 << " microseconds" << std::endl;
        // std::cout << "SIMD parallel gemm product took " << t_simd_parallel * 1000000 << " microseconds" << std::endl;
    }
}