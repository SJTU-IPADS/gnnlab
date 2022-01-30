FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

# change software source
COPY ./docker/sources.list /etc/apt/sources.list
COPY ./docker/.condarc /root/.condarc
COPY ./docker/pip.conf /root/.pip/pip.conf

# apt software
RUN apt-get update && apt-get install -y wget\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# miniconda
WORKDIR /app
ENV PATH="/miniconda3/bin:$PATH"
# installation
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda-latest.sh \
    && bash ./miniconda-latest.sh -b -p /miniconda3 \
    && ln -s /miniconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc \
    && find /miniconda3/ -follow -type f -name '*.a' -delete \
    && find /miniconda3/ -follow -type f -name '*.js.map' -delete \
    && conda clean -afy
# create environment
RUN conda create -n fgnn_env cmake cudnn==7.6.5 python==3.8 \
      pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch -y \
    && conda clean -afy \
    && echo "conda activate fgnn_env" >> ~/.bashrc

# Make RUN commands use the new environment:
SHELL ["conda", "run", "--no-capture-output", "-n", "fgnn_env", "/bin/bash", "-c"]

# install dgl
WORKDIR /app/source
COPY ./3rdparty/dgl ./dgl
RUN pushd ./dgl \
    && bash build_with_cudnn.sh \
    && pip install numpy>=1.14.0 scipy>=1.1.0 networkx>=2.1 requests>=2.19.0 \
    && bash install.sh \
    && rm -rf build \
    && popd

# install PyG
# XXX: command "pip install torch-scatter -f https://data.pyg.org/whl/torch-1.7.0+cu101.html" fail
#      use command "FORCE_CUDA=1 pip3 install --no-cache-dir --verbose torch-scatter==2.0.8" instead
RUN FORCE_CUDA=1 pip3 install --no-cache-dir --verbose torch-scatter==2.0.8 \
    && pip install torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.7.0+cu101.html \
    && pip install torch-geometric==2.0.1 \
    && pip install torch-cluster==1.5.9 -f https://data.pyg.org/whl/torch-1.7.0+cu101.html \
    && pip install torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.7.0+cu101.html

# install samgraph
COPY . ./samgraph
RUN pushd ./samgraph \
    && pushd utility/fastgraph \
    && python setup.py install \
    && popd \
    && python setup.py install \
    && rm -rf build \
    && rm -rf 3rdparty/dgl \
    && popd \
    && echo "ulimit -l unlimited" >> ~/.bashrc
