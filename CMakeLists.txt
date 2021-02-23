cmake_minimum_required(VERSION 3.9)

project(gpu-sampling C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O2 -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -fopenmp -g")

find_package(Torch REQUIRED)

include_directories(src)
include_directories(third_party/parallel-hashmap)
include_directories(third_party/json/single_include)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${TORCH_INCLUDE_DIRS})

add_subdirectory(third_party/gflags)

file(GLOB_RECURSE SOURCES "src/*.cu")

add_executable(gpu-system "src/system.cpp" ${SOURCES})
add_executable(data-server "src/data_server.cpp" ${SOURCES})
add_executable(sampling-cpu-eval "eval/sampling_cpu_eval.cpp" ${SOURCES})
add_executable(graphsage "eval/graphsage.cpp")
add_executable(torch-test "eval/torch_test.cpp")
add_executable(sample-block-test "eval/test_sample_block.cpp" ${SOURCES})
target_link_libraries(sampling-cpu-eval gflags::gflags ${TORCH_LIBRARIES})
target_link_libraries(graphsage gflags::gflags ${CUDA_cusparse_LIBRARY} ${TORCH_LIBRARIES})
target_link_libraries(torch-test ${TORCH_LIBRARIES})
target_link_libraries(sample-block-test gflags::gflags ${CUDA_cusparse_LIBRARY})
