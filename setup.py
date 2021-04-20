#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Package meta-data.
NAME = 'samgraph'
DESCRIPTION = 'A high-performance GPU-based graph sampler for deep graph learning application'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.0.1'

# What packages are required for this module to be executed?
REQUIRED = [
    # 'cffi>=1.4.0',
]


# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except OSError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION

# Where the magic happens:

setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(exclude=('tests',)),
    install_requires=REQUIRED,
    include_package_data=True,
    license='Apache',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Operating System :: POSIX :: Linux'
    ],
    ext_modules=[
        CUDAExtension(
            name='samgraph.torch.c_lib',
            sources=[
                'samgraph/common/common.cc',
                'samgraph/common/config.cc',
                'samgraph/common/engine.cc',
                'samgraph/common/extractor.cc',
                'samgraph/common/graph_pool.cc',
                'samgraph/common/logging.cc',
                'samgraph/common/operation.cc',
                'samgraph/common/profiler.cc',
                'samgraph/common/ready_table.cc',
                'samgraph/common/task_queue.cc',
                'samgraph/common/cpu/cpu_convert.cc',
                'samgraph/common/cpu/cpu_engine.cc',
                'samgraph/common/cpu/cpu_loops.cc',
                'samgraph/common/cpu/cpu_parallel_hashtable.cc',
                'samgraph/common/cpu/cpu_permutator.cc',
                'samgraph/common/cpu/cpu_sampling.cc',
                'samgraph/common/cpu/cpu_simple_hashtable.cc',
                'samgraph/common/cuda/cuda_convert.cc',
                'samgraph/common/cuda/cuda_engine.cc',
                'samgraph/common/cuda/cuda_hashtable.cu',
                'samgraph/common/cuda/cuda_kernels.cu',
                'samgraph/common/cuda/cuda_loops.cc',
                'samgraph/common/cuda/cuda_mapping.cu',
                'samgraph/common/cuda/cuda_permutator.cc',
                'samgraph/common/cuda/cuda_sampling.cu',
                'samgraph/torch/adapter.cc',
            ],
            include_dirs=[os.path.join(
                here, '3rdparty/cub'), os.path.join(here, '3rdparty/parallel-hashmap')],
            libraries=['cusparse'],
            extra_link_args=['-Wl,--version-script=samgraph.lds', '-fopenmp'],
            extra_compile_args={
                'cxx': ['-std=c++14', '-g', '-fopt-info',  '-fPIC',
                        '-Ofast',
                        # '-O0',
                        '-Wall', '-fopenmp', '-march=native'],
                'nvcc': ['-std=c++14', '-g', '-arch=sm_35', '--ptxas-options=-v', '--compiler-options', "'-fPIC'"]
            })
    ],
    # $ setup.py publish support.
    cmdclass={
        'build_ext': BuildExtension
    },
    # cffi is required for PyTorch
    # If cffi is specified in setup_requires, it will need libffi to be installed on the machine,
    # which is undesirable.  Luckily, `install` action will install cffi before executing build,
    # so it's only necessary for `build*` or `bdist*` actions.
    setup_requires=REQUIRED
)
