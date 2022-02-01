#!/bin/bash

HERE="$(dirname $(readlink -f $0))"

pushd "$HERE/fastgraph"
python setup.py install
popd