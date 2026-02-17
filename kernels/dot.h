#ifndef DOT_H
#define DOT_H

#include "../lib/vec_naive.h"

namespace kernels {
    float dot_simple(const naive::Vec& a, const naive::Vec& b);
    float dot_parallel(const naive::Vec& a, const naive::Vec& b, uint32_t num_threads);
}

#endif // DOT_H