#include "gemm.h"
#include <thread>
#include <vector>

namespace {
    void gemm_kernel(const Mat &a, const Mat &b, Mat &c, size_t start, size_t end)
    {
        for (size_t i = start; i < end; i++) {
            for(size_t j = 0; j < b.cols; j++) {
                c(i, j) = 0.0f;
                for(size_t k = 0; k < a.cols; k++)
                {
                    c(i, j) += a(i, k) * b(k, j);
                }
            }
        }
    }
}

namespace kernels {
    Mat gemm_simple(const Mat& a, const Mat& b) {
        Mat c(a.rows, b.cols);
        
        for(uint32_t i = 0; i < a.rows; i++) {
            for(uint32_t j = 0; j < b.cols; j++) {
                c(i, j) = 0.0f;
                for(uint32_t k = 0; k < a.cols; k++) {
                    c(i, j) += a(i, k) * b(k, j);
                }
            }
        }
        
        return c;
    }

    Mat gemm_omp(const Mat &a, const Mat &b) {
        Mat c(a.rows, b.cols);
        
        #pragma omp parallel for
        for(uint32_t i = 0; i < a.rows; i++) {
            for(uint32_t j = 0; j < b.cols; j++) {
                c(i, j) = 0.0f;
                for(uint32_t k = 0; k < a.cols; k++) {
                    c(i, j) += a(i, k) * b(k, j);
                }
            }
        }
        
        return c;
    }

    Mat gemm_parallel(const Mat &a, const Mat &b, uint32_t num_threads)
    {
        Mat c(a.rows, b.rows);

        std::vector<std::thread> threads(num_threads);

        for(uint32_t i = 0; i < num_threads; i++)
        {
            size_t start = i * a.rows / num_threads;
            size_t end = (i + 1) * a.rows / num_threads;
            threads[i] = std::thread(gemm_kernel, std::cref(a), std::cref(b), std::ref(c), start, end);
        }

        for(auto& t : threads)
            t.join();

        return c;
    }
}
