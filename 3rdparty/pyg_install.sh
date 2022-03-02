#!/bin/bash

# FORCE_CUDA=1 pip install --no-cache-dir --verbose torch-scatter==2.0.8 \
pip install torch-scatter==2.0.7 -f https://data.pyg.org/whl/torch-1.7.0+cu101.html \
    && pip install torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.7.0+cu101.html \
    && pip install torch-geometric==2.0.1 \
    && pip install torch-cluster==1.5.9 -f https://data.pyg.org/whl/torch-1.7.0+cu101.html \
    && pip install torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.7.0+cu101.html
