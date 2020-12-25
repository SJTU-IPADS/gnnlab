cmake_minimum_required(VERSION 3.9)

project(gpu-sampling C CXX CUDA)

include_directories(src)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

file(GLOB SOURCES "src/*.cc" "src/*.hpp" "src/*.cu")

add_executable(gpu-sampling ${SOURCES})
