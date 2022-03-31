# GNNLab

GNNLab is a factored system for sample-based GNN training over GPUs. GNNLab dedicates each GPU to the task of graph sampling or model training. It accelerates both tasks by eliminating GPU memory contention. Furthermore, GNNLab embodies a new pre-sampling based caching policy that takes both sampling algorithms and GNN datasets into account, showing an efficient and robust caching performance.

## Paper
GNNLab: A Factored System For Sample-based GNN Training Over GPUs, EuroSys'22, *Jianbang Yang, Dahai Tang, Xiaoniu Song, Lei Wang, Qiang Yin, Rong Chen, Wenyuan Yu, and Jingren Zhou.*

**Artifact Evaluation** for GNNLab: [Link](https://github.com/SJTU-IPADS/fgnn-artifacts).

## Terminology
FGNN is the initial version GNNLab, while SGNN is the initial version of T<sub>SOTA</sub>, a baseline system. SamGraph is the framework shared by the above system.

## Table of Contents
  - [Project Structure](#project-structure)
  - [Paper's Hardware Configuration](#papers-hardware-configuration)
  - [Installation](#installation)
    - [Software Version](#software-version)
    - [Install CUDA10.1](#install-cuda101)
    - [Install GCC-7](#install-gcc-7)
    - [Install GNN Training Systems](#install-gnn-training-systems)
    - [Change ULIMIT](#change-ulimit)
    - [Docker Support](#docker-support)
  - [Dataset Preprocessing](#dataset-preprocessing)
  - [QuickStart: Use FGNN to train GNN models](#quickstart-use-fgnn-to-train-gnn-models)
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
- 8 * NVIDIA V100 GPUs (16GB of memory each)
- 2 * Intel Xeon Platinum 8163 CPUs (24 cores each)
- 512GB RAM


## Installation

### Software Version

- Ubuntu 18.04 or Ubuntu 20.04
- gcc-7, g++-7
- CMake >= 3.14
- CUDA v10.1
- Python v3.8
- PyTorch v1.7.1
- DGL V0.7.1
- PyG v2.0.1

### Install CUDA10.1

FGNN is built on CUDA 10.1. Follow the instructions in https://developer.nvidia.com/cuda-10.1-download-archive-base to install CUDA 10.1, and make sure that `/usr/local/cuda` is linked to `/usr/local/cuda-10.1`.

### Install GCC-7

CUDA10.1 requires GCC (version<=7). Hence, make sure that `gcc` is linked to `gcc-7`, and `g++` is linked to `g++-7`. 

    ```bash
    # Ubuntu
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 7
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 7
    ```


### Install GNN Training Systems

We use conda to manage our python environment.

1. We use conda to manage our python environment.

    ```bash
    conda create -n fgnn_env python==3.8 pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch -y
    conda activate fgnn_env
    conda install cudnn numpy scipy networkx tqdm pandas ninja cmake -y # System cmake is too old to build DGL
    sudo apt install gnuplot # Install gnuplot for experiments:
    ```


2. Download GNN systems.

    ```bash
    # Download FGNN source code
    git clone --recursive https://github.com/SJTU-IPADS/fgnn-artifacts.git
    ```

3. Install DGL, PyG, and FastGraph. The package FastGraph is used to load datasets for GNN systems in all experiments.

    ```bash
    # Install DGL
    ./fgnn-artifacts/3rdparty/dgl_install.sh

    # Install PyG
    ./fgnn-artifacts/3rdparty/pyg_install.sh

    # Install fastgraph
    ./fgnn-artifacts/utility/fg_install.sh
    ```

    

4. Install FGNN (also called SamGraph) and SGNN.
   
    ```bash
    cd fgnn-artifacts
    ./build.sh
    ```

### Change ULIMIT
Both DGL and FGNN need to use a lot of system resources. DGL CPU sampling requires cro-processing communications while FGNN's global queue requires memlock(pin) memory to enable faster memcpy between host memory and GPU memory. Hence we have to set the user limit.


Append the following content to `/etc/security/limits.conf` and then `reboot`:

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

### Docker Support
We provide two methods to get the Docker image.

1. Build the image from Dockerfile

    We provide a Dockerfile to build the experiment image. The file is in the root directory of this repository. Users can use the following command to create a Docker environment.
    ```bash
    docker build -t gnnlab/fgnn:v1.0 .
    ```

2. Pull the image from Docker Hub (about 4.24GB)
    ```
    docker pull gnnlab/fgnn:v1.0
    # We only test this image in the platform with all V100 GPUs
    # If users meet some running problems on other GPUs, we provide all tools needed in the image to re-build DGL, FGNN and PyG.
    ```


Then users can run tests in Docker.
```bash
docker run --ulimit memlock=-1 --rm --gpus all -v $HOST_VOLUMN:/graph-learning -it gnnlab/fgnn:v1.0 bash
```

**Make sure that Docker can support CUDA while building images. Here is a [reference](https://stackoverflow.com/questions/59691207) to solve Docker building images with CUDA support.**


## Dataset Preprocessing

See [`datagen/README.md`](datagen/README.md) to find out how to preprocess datasets.



## QuickStart: Use FGNN to train GNN models

FGNN is compiled into Python library. We have written several GNN models using FGNN’s APIs. These models are in `fgnn-artifacts/example` and are easy to run as following:

```bash
cd fgnn-artifacts/example

python samgraph/multi_gpu/train_gcn.py --dataset papers100M --num-train-worker 1 --num-sample-worker 1 --pipeline --cache-policy pre_sample --cache-percentage 0.1 --num-epoch 10 --batch-size 8000
```



## Experiments

Our experiments have been automated by scripts (`run.py`). Each figure or table in our paper is treated as one experiment and is associated with a subdirectory in `fgnn-artifacts/exp`. The script will automatically run the experiment, save the logs into files, and parse the output data from the files.

Note that running all experiments may take several hours. This [table](exp/README.md#expected-running-time) lists the expected running time for each experiment.

See [`exp/README.md`](exp/README.md) to find out how to conduct the experiments.



## License

FGNN is released under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0.html).




## Academic and Conference Papers

[**EuroSys**] GNNLab: A Factored System for Sample-based GNN Training over GPUs. Jianbang Yang, Dahai Tang, Xiaoniu Song, Lei Wang, Qiang Yin, Rong Chen, Wenyuan Yu, Jingren Zhou. Proceedings of the 17th European Conference on Computer Systems, Rennes, France, April, 2022.
