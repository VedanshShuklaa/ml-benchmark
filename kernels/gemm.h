#ifndef GEMM_H
#define GEMM_H

#include "../lib/mat.h"
#include <cstdint>

namespace kernels {
    Mat gemm_simple(const Mat& a, const Mat& b);
    Mat gemm_parallel(const Mat& a, const Mat& b, uint32_t num_threads);
    Mat gemm_simd(const Mat& a, const Mat& b);
    Mat gemm_simd_parallel(const Mat& a, const Mat& b, uint32_t num_threads);
    Mat gemm_omp(const Mat& a, const Mat& b);
}

#endif // GEMM_H