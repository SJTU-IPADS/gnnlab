cmake_minimum_required(VERSION 3.20)

project(cuda-memory-demo LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 70)

add_executable(race-write race-write.cu)

# target_compile_options(race-write PRIVATE 
#     $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_70,code=sm_70>)
