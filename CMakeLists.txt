cmake_minimum_required(VERSION 3.16)
project(ml_concurrency_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(mlbench_lib
    lib/vec.cpp
)

target_include_directories(mlbench_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

add_library(mlbench_kernels
    kernels/dot.cpp
)

target_include_directories(mlbench_kernels
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)

add_executable(mlbench
    main.cpp
    bench/bench_dot.cpp
)

target_include_directories(mlbench
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/bench
)

target_link_libraries(mlbench
    PRIVATE
        mlbench_lib
        mlbench_kernels
)

find_package(Threads REQUIRED)
target_link_libraries(mlbench PRIVATE Threads::Threads)
target_link_libraries(mlbench_lib PRIVATE Threads::Threads)
target_link_libraries(mlbench_kernels PRIVATE Threads::Threads)
