#include <cstdint>
#include "../lib/vec_naive.h"
#include "dot.h"

namespace kernels {
    float dot_simple(const naive::Vec& a, const naive::Vec& b) {
        float result = 0.0f;
        for(uint32_t i = 0; i < a.size(); i++) {
            result += a[i] * b[i];
        }
        return result;
    }
}
