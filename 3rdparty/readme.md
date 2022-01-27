# Build & Install DGL

First apply our patch:
```bash
cd 3rdparty/dgl
git apply ../dgl.patch
```

Then build dgl:
```bash
export CUDNN_LIBRARY=$CONDA_PREFIX/lib
export CUDNN_LIBRARY_PATH=$CONDA_PREFIX/lib
export CUDNN_ROOT=$CONDA_PREFIX
export CUDNN_INCLUDE_DIR=$CONDA_PREFIX/include
export CUDNN_INCLUDE_PATH=$CONDA_PREFIX/include
cmake -S . -B build -DUSE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME -DBUILD_TORCH=ON -DCMAKE_BUILD_TYPE=Release
pushd build
make -j
popd build
```

If you want to specify path of cuda toolkit or you do not have `/usr/local/cuda`, then you must pass `-DCUDA_TOOLKIT_ROOT_DIR=<path to your cuda>` to cmake above.

Lastly, install dgl:
```bash
pushd python
python setup.py install
popd
```