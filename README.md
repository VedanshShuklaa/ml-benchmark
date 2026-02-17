# ml-concurrency-bench

This is my tiny, educational C++ bench project for learning how concurrency and low-level choices affect ML-style kernels (vectors, dot products, etc.).

Right now (what’s actually in the repo)

* a very small `naive::Vec` (simple heap array + indexing)
* a single-threaded dot product (`kernels::dot_simple`) — the plain baseline
* a multi-threaded dot product (`kernels::dot_parallel`) that partitions the vector and reduces per-thread partial sums
* a tiny timing helper `time_function` that does a warmup + averages a few runs
* a small bench harness (`bench_naive()` called from `bench::bench_dot()`)

---

# What I want from this repo (my personal goals)

* learn the difference between single-threaded and multi-threaded dot products
* understand false sharing, cache effects, and how chunking/partitioning matters
* get a feel for what micro-optimizations actually move the needle (alignment, vectorization, padding, thread pool vs raw `std::thread`)
* understand how to write clean, concise and performant code
* understand how to measure and analyze performance

---

# How to build

If you don’t have a CMake set up yet, this single-command compile works for experimenting:

```bash
# from repo root (example)
g++ -O3 -std=c++17 src/*.cpp lib/*.cpp -pthread -o mlbench
```

If you use CMake, then

```bash
mkdir -p build && cd build
cmake ..
make
```
