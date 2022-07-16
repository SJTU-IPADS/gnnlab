cmake_minimum_required(VERSION 3.14)
SITE_NAME(HOSTNAME)
if(${HOSTNAME} STREQUAL "j63c09243.sqa.eu95")
set(CMAKE_CUDA_COMPILER /usr/local/cuda-10.1/bin/nvcc)
elseif(${HOSTNAME} STREQUAL "meepo4")
set(CMAKE_CUDA_COMPILER /usr/local/cuda-10.2/bin/nvcc)
elseif(${HOSTNAME} STREQUAL "meepo5")
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11/bin/nvcc)
elseif(${HOSTNAME} STREQUAL "SH-IDC1-10-140-0-192")
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.7/bin/nvcc)
else()
message( STATUS "host is $ENV{HOSTNAME}")
message( WARNING "unknown host" )
endif()

project(samgraph_cpp_version C CXX CUDA)

# set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_BUILD_TYPE "Debug")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(COMMON_FLAGS " -O0 -g -DSXN_REVISED ")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
  set(COMMON_FLAGS " -O3 -g -DSXN_REVISED ")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  ${COMMON_FLAGS} -fPIC -Wall -fopenmp -march=native -D_GLIBCXX_USE_CXX11_ABI=0 -L/opt/gurobi-install/gurobi951/linux64/lib")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${COMMON_FLAGS} --compiler-options '-fPIC -fopenmp -D_GLIBCXX_USE_CXX11_ABI=0' -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80")
message(DEBUG "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(DEBUG "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
include_directories(
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  # ../3rdparty/cub
  ../../3rdparty/parallel-hashmap
  ../../3rdparty/CLI11/include
  ../../samgraph
  /opt/intel/oneapi/tbb/latest/include
  $ENV{GUROBI_HOME}/include
)
link_directories(
  $ENV{CONDA_PREFIX}/lib
  $ENV{GUROBI_HOME}/lib
  /opt/intel/oneapi/tbb/latest/lib/intel64/gcc4.8
  /usr/local/cuda/lib64
)


file(GLOB_RECURSE SAMGRAPH_SRC
  ../../samgraph/common/coll_cache/*
  ../../samgraph/common/common.cc
  ../../samgraph/common/common.cc
  ../../samgraph/common/constant.cc
  ../../samgraph/common/logging.cc
  ../../samgraph/common/device.cc
  ../../samgraph/common/run_config.cc
  ../../samgraph/common/cpu/mmap_cpu_device.cc
  ../../samgraph/common/cuda/cuda_device.cc
  ../../samgraph/common/cuda/cuda_utils.cu
  ../../samgraph/common/cpu/cpu_device.cc
  ../../samgraph/common/workspace_pool.cc
  common/*.cc
  common/*.cu)

# build samgraph library

add_executable(
  optimal_solve
  ${SAMGRAPH_SRC}
  solve_func.cu
  main.cc
)

target_link_libraries(
  optimal_solve
  rt
  tbb
  cuda
  cudart
  gurobi_c++
  # gurobi_g++5.2
  gurobi95
  cusparse
)