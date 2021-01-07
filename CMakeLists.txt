cmake_minimum_required(VERSION 3.9)

project(gpu-sampling C CXX CUDA)

set(CMAKE_CXX_FLAGS "-O2 -fopenmp")

include_directories(src)
include_directories(third_party/parallel-hashmap)
include_directories(third_party/json/single_include)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_subdirectory(third_party/gflags)

file(GLOB_RECURSE SOURCES "src/*.cu")

add_executable(gpu-system "src/system.cpp" ${SOURCES})
add_executable(data-server "src/data_server.cpp" ${SOURCES})
add_executable(sampling-cpu-eval "eval/sampling_cpu_eval.cpp" ${SOURCES})
target_link_libraries(sampling-cpu-eval gflags::gflags)
