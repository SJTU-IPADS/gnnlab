cmake_minimum_required(VERSION 3.14)
SITE_NAME(HOSTNAME)
if(${HOSTNAME} STREQUAL "j63c09243.sqa.eu95")
set(CMAKE_CUDA_COMPILER /usr/local/cuda-10.1/bin/nvcc)
elseif(${HOSTNAME} STREQUAL "meepo4")
set(CMAKE_CUDA_COMPILER /usr/local/cuda-10.2/bin/nvcc)
else()
message( STATUS "host is $ENV{HOSTNAME}")
message( FATAL_ERROR "unknown host" )
endif()

project(samgraph_cpp_version CXX CUDA)

set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(COMMON_FLAGS " -O0 -g -DSXN_REVISED ")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
  set(COMMON_FLAGS " -O3 -g -DSXN_REVISED " )
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  ${COMMON_FLAGS} -fPIC -Wall -fopenmp -march=native -D_GLIBCXX_USE_CXX11_ABI=0")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${COMMON_FLAGS} --compiler-options '-fPIC -D_GLIBCXX_USE_CXX11_ABI=0' -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_70,code=sm_70")
include_directories(
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ../3rdparty/cub
  ../3rdparty/parallel-hashmap
  ../3rdparty/CLI11/include
  $ENV{CONDA_PREFIX}/lib/python3.8/site-packages/torch/include
  $ENV{CONDA_PREFIX}/lib/python3.8/site-packages/torch/include/torch/csrc/api/include
  $ENV{CONDA_PREFIX}/include/python3.8)
link_directories(
  $ENV{CONDA_PREFIX}/lib
  $ENV{CONDA_PREFIX}/lib/python3.8/site-packages/torch/lib
  /usr/local/cuda/lib64
)


set(SAMGRAPH_SRC 
  common/common.cc
  common/constant.cc
  common/device.cc
  common/engine.cc
  common/graph_pool.cc
  common/logging.cc
  common/operation.cc
  common/profiler.cc
  common/run_config.cc
  common/task_queue.cc
  common/workspace_pool.cc
  common/memory_queue.cc
  common/cpu/cpu_device.cc
  common/cpu/cpu_engine.cc
  common/cpu/cpu_extraction.cc
  common/cpu/cpu_hashtable0.cc
  common/cpu/cpu_hashtable1.cc
  common/cpu/cpu_hashtable2.cc
  common/cpu/cpu_loops_arch0.cc
  common/cpu/cpu_loops.cc
  common/cpu/cpu_random.cc
  common/cpu/cpu_sampling_khop0.cc
  common/cpu/cpu_sampling_khop1.cc
  common/cpu/cpu_sampling_khop2.cc
  common/cpu/cpu_sampling_random_walk.cc
  common/cpu/cpu_sampling_weighted_khop.cc
  common/cpu/cpu_sanity_check.cc
  common/cpu/cpu_shuffler.cc
  common/cpu/mmap_cpu_device.cc
  common/cuda/cuda_cache.cu
  common/cuda/cuda_cache_manager_device.cu
  common/cuda/cuda_cache_manager_host.cc
  common/cuda/cuda_device.cc
  common/cuda/cuda_engine.cc
  common/cuda/cuda_extract_neighbour.cu
  common/cuda/cuda_extraction.cu
  common/cuda/cuda_frequency_hashmap.cu
  common/cuda/cuda_hashtable.cu
  common/cuda/cuda_loops_arch1.cc
  common/cuda/cuda_loops_arch2.cc
  common/cuda/cuda_loops_arch3.cc
  common/cuda/cuda_loops_arch4.cc
  common/cuda/cuda_loops_arch7.cc
  common/cuda/cuda_loops.cc
  common/cuda/cuda_mapping.cu
  common/cuda/cuda_random_states.cu
  common/cuda/cuda_sampling_khop0.cu
  common/cuda/cuda_sampling_khop1.cu
  common/cuda/cuda_sampling_khop2.cu
  common/cuda/cuda_sampling_random_walk.cu
  common/cuda/cuda_sampling_weighted_khop.cu
  common/cuda/cuda_sampling_weighted_khop_prefix.cu
  common/cuda/cuda_sampling_weighted_khop_hash_dedup.cu
  common/cuda/cuda_sanity_check.cu
  common/cuda/cuda_shuffler.cc
  common/cuda/pre_sampler.cc
  common/dist/dist_engine.cc
  common/dist/dist_loops.cc
  common/dist/dist_loops_arch5.cc
  common/dist/dist_loops_arch6.cc
  common/dist/dist_cache_manager_device.cu
  common/dist/dist_cache_manager_host.cc
  common/dist/pre_sampler.cc
  common/dist/dist_shuffler.cc
  common/dist/dist_shuffler_aligned.cc
)

# add_library(samgraph SHARED
#   ${SAMGRAPH_SRC}
# )

add_executable(
  samgraph_cpp_no_train_link_torch
  ${SAMGRAPH_SRC}
  torch/adapter.cc
  main.cc
)

add_executable(
  samgraph_cpp_no_train
  ${SAMGRAPH_SRC}
  main.cc
)
target_link_libraries(
  samgraph_cpp_no_train
  # samgraph
)
target_link_libraries(
  samgraph_cpp_no_train_link_torch
  # samgraph
  cusparse
  c10
  c10_cuda
  cudart
  torch
  torch_cpu
  torch_python
  torch_global_deps
  torch_cuda
  python3.8
)