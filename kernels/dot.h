#ifndef DOT_H
#define DOT_H

#include "../lib/vec.h"

namespace kernels {
    float dot_simple(const Vec& a, const Vec& b);
    float dot_parallel(const Vec& a, const Vec& b, uint32_t num_threads);
    float dot_simd(const Vec& a, const Vec& b);
    float dot_simd_parallel(const Vec& a, const Vec& b, uint32_t num_threads);
}

#endif // DOT_H