#include "gemm.h"
#include <thread>
#include <vector>
#include <immintrin.h>
#include <pmmintrin.h>
#include <xmmintrin.h>

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

    void gemm_simd_kernel(const Mat &a, const Mat &b, const Mat &bt, uint32_t start, uint32_t end, Mat &c) {
        // Just a reminder to myself: I accidentally added this line here and it shot up the times from 13k microseconds to 90k microseconds
        // Mat bt = b.transpose();
        
        for(uint32_t i = start; i < end; i++) {
            for(uint32_t j = 0; j < b.cols; j++) {
                __m128 sumrc = _mm_setzero_ps();
                for(uint32_t k = 0; k + 3 < a.cols; k += 4) {
                    __m128 av = _mm_load_ps(&a(i, k));
                    __m128 bv = _mm_load_ps(&bt(j, k));
                    sumrc = _mm_add_ps(sumrc, _mm_mul_ps(av, bv));
                }

                sumrc = _mm_hadd_ps(sumrc, sumrc);
                sumrc = _mm_hadd_ps(sumrc, sumrc);

                for (uint32_t k_tail = (a.cols / 4) * 4; k_tail < a.cols; ++k_tail) {
                    sumrc[0] += a(i, k_tail) * b(k_tail, j);
                }

                c(i, j) = _mm_cvtss_f32(sumrc);
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

    Mat gemm_simd(const Mat &a, const Mat &b) {    
        Mat c(a.rows, b.cols);

        Mat bt = b.transpose();
        
        for(uint32_t i = 0; i < a.rows; i++) {
            for(uint32_t j = 0; j < b.cols; j++) {
                __m128 sumrc = _mm_setzero_ps();
                for(uint32_t k = 0; k + 3 < a.cols; k += 4) {
                    __m128 av = _mm_load_ps(&a(i, k));
                    __m128 bv = _mm_load_ps(&bt(j, k));
                    sumrc = _mm_add_ps(sumrc, _mm_mul_ps(av, bv));
                }

                sumrc = _mm_hadd_ps(sumrc, sumrc);
                sumrc = _mm_hadd_ps(sumrc, sumrc);

                for (uint32_t k_tail = (a.cols / 4) * 4; k_tail < a.cols; ++k_tail) {
                    sumrc[0] += a(i, k_tail) * b(k_tail, j);
                }

                c(i, j) = _mm_cvtss_f32(sumrc);
            }
        }
        
        return c;
    }

    Mat gemm_simd_parallel(const Mat &a, const Mat &b, uint32_t num_threads) {
        Mat c(a.rows, b.cols);
        Mat bt = b.transpose();
        
        std::vector<std::thread> threads(num_threads);
        
        for (uint32_t i = 0; i < num_threads; i++)
        {
            uint32_t start = i * a.rows / num_threads;
            uint32_t end = (i + 1) * a.rows / num_threads;
            threads[i] = std::thread(gemm_simd_kernel, std::cref(a), std::cref(b), std::cref(bt), start, end, std::ref(c));
        }
        
        for(auto& t : threads)
            t.join();
        
        return c;
    }

    Mat gemm_parallel(const Mat &a, const Mat &b, uint32_t num_threads)
    {
        Mat c(a.rows, b.cols);

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
