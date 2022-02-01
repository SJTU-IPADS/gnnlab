#!/bin/bash

HERE="$(dirname $(readlink -f $0))"

pushd "$HERE"
python setup.py install
popd