#include "bench.h"
#include <thread>

constexpr uint64_t SIZE = 1e7;

void bench_naive()
{
    naive::Vec a(SIZE, 1.0f);
    naive::Vec b(SIZE, 1.0f);
    double t_simple = time_function([&] {
        kernels::dot_simple(a, b);
    });

    uint32_t num_threads = std::thread::hardware_concurrency();
    double t_parallel = time_function([&] {
        kernels::dot_parallel(a, b, num_threads);
    });
    
    std::cout << "Simple and naive dot product took " << t_simple << " seconds" << std::endl;
    std::cout << "Parallel and naive dot product took " << t_parallel << " seconds" << std::endl;
}


namespace bench {
    void bench_dot() {
        bench_naive();
    }
}