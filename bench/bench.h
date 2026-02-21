#ifndef BENCH_H
#define BENCH_H

#include "../kernels/dot.h"
#include "../kernels/gemm.h"
#include "../utils/timer.h"

namespace bench {
    void bench_dot();
    void bench_gemm();
}

#endif // BENCH_H