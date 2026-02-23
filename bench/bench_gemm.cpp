#include "bench.h"
#include <thread>
#include <cassert>
#include <random>
#include <cmath>

constexpr uint64_t SIZE = 512;
constexpr float EPSILON = 1e-4;

namespace bench {
    inline bool nearly_equal(float a, float b)
    {
        return std::abs(a - b) < EPSILON;
    }

    inline bool matrices_equal(const Mat& a, const Mat& b)
    {
        if (a.rows != b.rows || a.cols != b.cols) {
            return false;
        }
        for (uint64_t i = 0; i < a.rows; i++) {
            for (uint64_t j = 0; j < a.cols; j++) {
                if (!nearly_equal(a(i, j), b(i, j))) {
                    return false;
                }
            }
        }
        return true;
    }

    void fill_random(Mat& m) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        for (uint64_t i = 0; i < m.rows; i++) {
            for (uint64_t j = 0; j < m.cols; j++) {
                m(i, j) = dis(gen);
            }
        }
    }

    void bench_gemm() {
        Mat a(SIZE, SIZE);
        Mat b(SIZE, SIZE);

        uint32_t num_threads = std::thread::hardware_concurrency();

        fill_random(a);
        fill_random(b);

        Mat c_simple        = kernels::gemm_simple(a, b);
        Mat c_parallel      = kernels::gemm_parallel(a, b, num_threads);
        Mat c_omp           = kernels::gemm_omp(a, b);
        Mat c_simd          = kernels::gemm_simd(a, b);
        Mat c_simd_parallel = kernels::gemm_simd_parallel(a, b, num_threads);

        // Checks to see if all other algorithms compute values properly
        assert(matrices_equal(c_simple, c_parallel));
        assert(matrices_equal(c_simple, c_omp));
        assert(matrices_equal(c_simple, c_simd));
        assert(matrices_equal(c_simple, c_simd_parallel));

        double t_simple = time_function([&] {
            kernels::gemm_simple(a, b);
        });

        double t_parallel = time_function([&] {
            kernels::gemm_parallel(a, b, num_threads);
        });

        double t_omp = time_function([&] {
            kernels::gemm_omp(a, b);
        });

        double t_simd = time_function([&] {
            kernels::gemm_simd(a, b);
        });
        
        double t_simd_parallel = time_function([&] {
            kernels::gemm_simd_parallel(a, b, num_threads);
        });
        
        std::cout << "Simple gemm product took " << t_simple * 1000000 << " microseconds" << std::endl;

        std::cout << "Parallel gemm product took " << t_parallel * 1000000 << " microseconds" << std::endl;

        std::cout << "OMP Parallel gemm product took " << t_omp * 1000000 << " microseconds" << std::endl;

        std::cout << "SIMD gemm product took " << t_simd * 1000000 << " microseconds" << std::endl;
        
        std::cout << "SIMD parallel gemm product took " << t_simd_parallel * 1000000 << " microseconds" << std::endl;
    }
}