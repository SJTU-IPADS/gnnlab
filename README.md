# SamGraph

SamGraph is a high-performance GPU-based graph sampler for deep graph learning application

## Get Started

1. We recommend a test environment installation from conda.
2. Install Python, CUDA, and Pytorch environment from conda:  
    ```bash
    conda create -n fgnn_env python==3.8 pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch -y
    conda deactivate
    conda activate fgnn_env
    ```
3. Download the FGNN(SamGraph) source code, install DGL and fastgraph in the source tree:
    ```bash
    pushd 3rdparty/dgl
    bash build.sh
    bash install.sh
    popd
    pushd utility/fastgraph
    python setup.py install
    popd
    ```
   The package "fastgraph" is used for DGL's dataset processing.
4. Install FGNN(SamGraph):
    ```bash
    python setup.py install
    ```
5. Install PyG [TODO]

## Dataset preprocess
