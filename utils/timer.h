#include <chrono>

template<typename F>
double time_function(F&& fn, int runs = 5)
{
    // warmup
    fn();

    double total = 0.0;

    for (int i = 0; i < runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        fn();
        auto end = std::chrono::high_resolution_clock::now();

        total += std::chrono::duration<double>(end - start).count();
    }

    return total / runs;
}