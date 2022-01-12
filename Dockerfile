FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
SHELL ["/bin/bash", "-c"]
COPY ./docker/sources.list /etc/apt/sources.list
COPY ./docker/.condarc /root/.condarc
WORKDIR /app
COPY ./docker/miniconda-latest.sh .
ENV PATH="/miniconda3/bin:$PATH"
RUN bash ./miniconda-latest.sh -b -p /miniconda3 \
    && /miniconda3/bin/conda init bash \
    && source /root/.bashrc \
    && conda create -n fgnn_env cudnn==7.6.5 python==3.8 pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch -y
RUN conda init bash\
    && bash /root/.bashrc \
    && . /root/.bashrc
RUN conda init bash\
    && . /root/.bashrc \
    && conda activate fgnn_env \
    && conda clean -y -a

WORKDIR /app/source
COPY . .  #  copy source code
RUN conda init \
    && . /root/.bashrc \
    && conda activate fgnn_env \
    && pushd ./3rdparty/dgl \
    && bash install.sh \
    && popd \
    && pushd utility/fastgraph \
    && python setup.py install \
    && popd \
    && python setup.py install

