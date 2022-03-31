#!/bin/bash

HERE="$(dirname $(readlink -f $0))"

echo $HERE

pushd "$HERE/dgl"

git apply ../dgl.patch # patching for dataset loading

export CUDNN_LIBRARY=$CONDA_PREFIX/lib
export CUDNN_LIBRARY_PATH=$CONDA_PREFIX/lib
export CUDNN_ROOT=$CONDA_PREFIX
export CUDNN_INCLUDE_DIR=$CONDA_PREFIX/include
export CUDNN_INCLUDE_PATH=$CONDA_PREFIX/include
cmake -S . -B build -DUSE_CUDA=ON -DBUILD_TORCH=ON -DCMAKE_BUILD_TYPE=Release

pushd build
make -j
popd

pushd python
python setup.py install
popd

popd
