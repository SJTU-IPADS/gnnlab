#!/bin/bash

if [ ! -d "build" ]; then
    mkdir build
fi

cmake -S . -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build
