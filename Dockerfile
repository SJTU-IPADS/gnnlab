FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
COPY ./docker/sources.list /etc/apt/sources.list
COPY ./docker/.condarc /root/.condarc
WORKDIR /app
# miniconda
COPY ./docker/miniconda-latest.sh .
ENV PATH="/miniconda3/bin:$PATH"
RUN bash ./miniconda-latest.sh -b -p /miniconda3 \
    && ln -s /miniconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc \
    && find /miniconda3/ -follow -type f -name '*.a' -delete \
    && find /miniconda3/ -follow -type f -name '*.js.map' -delete \
    && conda clean -afy

RUN conda create -n fgnn_env cmake cudnn==7.6.5 python==3.8 \
      pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch -y \
    && conda clean -afy

# software
# RUN apt-get update && apt-get install -y cmake\
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Make RUN commands use the new environment:
SHELL ["conda", "run", "--no-capture-output", "-n", "fgnn_env", "/bin/bash", "-c"]

WORKDIR /app/source
COPY . .
# for pip source
COPY ./docker/pip.conf /root/.pip/pip.conf
RUN pushd ./3rdparty/dgl \
    && export \
    && bash build_with_cudnn.sh \
    && bash install.sh \
    && popd \
    && pushd utility/fastgraph \
    && python setup.py install \
    && popd \
    && python setup.py install

