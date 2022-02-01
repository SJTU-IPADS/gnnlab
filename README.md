# FGNN: A Factored System For Sample-based GNN Training Over GPUs

FGNN (previously named SamGraph) is a factored system for sample-based GNN training over GPUs, where each GPU is dedicated to the task of graph sampling or model training. It accelerates both tasks by eliminating GPU memory contention. Furthermore, FGNN embodies a new pre-sampling based caching policy that takes both sampling algorithms and GNN datasets into account, showing an efficient and robust caching performance.

- [FGNN: A Factored System For Sample-based GNN Training Over GPUs](#fgnn-a-factored-system-for-sample-based-gnn-training-over-gpus)
  - [Project Structure](#project-structure)
  - [Paper's Hardware Configuration](#papers-hardware-configuration)
  - [Installation](#installation)
    - [Software Version](#software-version)
    - [gcc-7 And CUDA10.1 Environment](#gcc-7-and-cuda101-environment)
    - [FGNN, DGL and PyG Environment](#fgnn-dgl-and-pyg-environment)
    - [Setting ulimit](#setting-ulimit)
  - [Dataset Preprocessing](#dataset-preprocessing)
  - [Quickstart Example](#quickstart-example)
  - [Experiments](#experiments)
  - [License](#license)
  - [Academic and Conference Papers](#academic-and-conference-papers)


## Project Structure

```bash
> tree .
├── datagen                     # Dataset Preprocessing
├── example
│   ├── dgl
│   │   ├── multi_gpu           # DGL models
│   ├── pyg
│   │   ├── multi_gpu           # PyG models
│   ├── samgraph
│   │   ├── balance_switcher    # FGNN Dynamic Switch
│   │   ├── multi_gpu           # FGNN models
│   │   ├── sgnn                # SGNN models
│   │   ├── sgnn_dgl            # DGL PinSAGE models(SGNN simulated)
├── exp                         # Experiment Scripts
│   ├── figXX
│   ├── tableXX
├── samgraph                    # FGNN, SGNN source codes
└── utility                     # Useful tools for dataset preprocessing
```



## Paper's Hardware Configuration
- 8x16GB NVIDIA V100 GPUs
- 2x24 cores Intel Xeon Platinum CPUs
- 512GB RAM

**In the AE environment we provided,  each V100 GPU has 32GB memory.**



## Installation

**We have already created accounts and setup out-of-the-box environments for AE reviewers. AE reviewers don't need to perform the following steps if AE reviewers choose to run the experiments on the machine we provided.**

**The AE machine and account information can be found in the AE appendix and the AE website comments.**

### Software Version

- Ubuntu 18.04 or Ubuntu 20.04
- Python v3.8
- PyTorch v1.7.1
- CUDA v10.1
- DGL V0.7.1
- PyG v2.0.1
- gcc-7 && g++-7
- CMake >= 3.14

### gcc-7 And CUDA10.1 Environment

1. Install CUDA 10.1. FGNN is built on CUDA 10.1. Follow the instructions in https://developer.nvidia.com/cuda-10.1-download-archive-base to install CUDA 10.1. Make sure that `/usr/local/cuda` is linked to `/usr/local/cuda-10.1`.

2. CUDA10.1 requires GCC version<=7. Make sure that `gcc` is linked to `gcc-7` and `g++` is linked to `g++-7`. 

    ```bash
    # Ubuntu
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 7
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 7
    ```


### FGNN, DGL and PyG Environment

We use conda to manage our python environment.

1. Install Python=3.8, cudatoolkit=10.1, and Pytorch=1.7.1 environment from conda: 

    ```bash
    conda create -n fgnn_env python==3.8 pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch -y
    conda activate fgnn_env
    conda install cudnn numpy scipy networkx tqdm pandas ninja cmake -y # System cmake is too old to build DGL

    ```
    Install gnuplot for experiments:
    ```bash
    # Ubuntu
    sudo apt install gnuplot
    ```


2. Download the FGNN source code, install DGL(See [`3rdparty/readme.md`](3rdparty/readme.md)) and fastgraph in the source. FGNN uses DGL as the training backend. The package "fastgraph" is used to load dataset for DGL and PyG in experiments.

    ```bash
    # Download FGNN source code
    git clone --recursive https://github.com/SJTU-IPADS/fgnn-artifacts.git
    
    # Install DGL
    ./fgnn-artifacts/3rdparty/dgl_install.sh

    # Install fastgraph
    ./fgnn-artifacts/utility/fg_install.sh

    # Install PyG for experiments
    ./fgnn-artifacts/3rdparty/pyg_install.sh
    ```

    

3. Install FGNN(Samgraph):
   
    ```bash
    cd fgnn-artifacts
    ./build.sh
    ```

### Setting ulimit
DGL CPU sampling requires cro-processing communications.FGNN global queue requires memlock to enable fast memcpy between host memory and GPU memory. So we have to set the user limit.


Add the following content to `/etc/security/limits.conf` and then `reboot`:

```bash
* soft nofile 65535         # for DGL CPU sampling
* hard nofile 65535         # for DGL CPU sampling
* soft memlock 200000000    # for FGNN global queue
* hard memlock 200000000    # for FGNN global queue
```

After reboot you can see:

```bash
> ulimit -n
65535

> ulimit -l
200000000
```



## Dataset Preprocessing

**AE reviewers don't need to perform the following steps if AE reviewers choose to run the experiments on the machine we provided. We have already downloaded and processed the dataset(`/graph-learning/samgraph`)**.

See [`datagen/README.md`](datagen/README.md).



## Quickstart Example

```bash
cd fgnn-artifacts/example/samgraph/multi_gpu

python train_gcn.py --dataset papers100M --num-train-worker 1 --num-sample-worker 1 --pipeline --cache-policy pre_sample --cache-percentage 0.1 --num-epoch 10 --batch-size 8000
```



## Experiments

See [`exp/README.md`](exp/README.md).



## License

FGNN is released under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0.html).




## Academic and Conference Papers

[**EuroSys**] FGNN: A Factored System for Sample-based GNN Training over GPUs. Jianbang Yang, Dahai Tang, Xiaoniu Song, Lei Wang, Qiang Yin, Rong Chen, Wenyuan Yu, Jingren Zhou. Proceedings of the 17th European Conference on Computer Systems, Rennes, France, April, 2022.
