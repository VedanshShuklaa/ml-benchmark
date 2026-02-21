#include "bench/bench.h"
#include <string>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        return 1;
    }
    
    if (std::string(argv[1]) == "dot") {
        bench::bench_dot();
    }
    else if (std::string(argv[1]) == "gemm") {
        bench::bench_gemm();
    }
    else {
        return 1;
    }
    
    return 0;
}