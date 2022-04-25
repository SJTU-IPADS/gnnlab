cmake_minimum_required(VERSION 3.20)

project(cuda-memory-demo LANGUAGES CXX CUDA)

add_executable(bandwidth-latency-test bandwidth-latency.cu utility.cu)

target_compile_features(bandwidth-latency-test PUBLIC cuda_std_17)

add_executable(simulate simulate.cu utility.cu)

add_executable(tmp tmp.cu utility.cu)

