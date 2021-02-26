#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import sys
import re
import shutil
from shutil import rmtree
import textwrap
import shlex
import subprocess

from setuptools import find_packages, setup, Command, Extension
from setuptools.command.build_ext import build_ext
from distutils.errors import CompileError, DistutilsError, DistutilsPlatformError, LinkError, DistutilsSetupError
from distutils import log as distutils_logger
from distutils.version import LooseVersion
import traceback

sampler_lib = Extension('samgraph.sampler.c_lib', [])
pytorch_lib = Extension('samgraph.torch.c_lib', [])

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


def is_build_action():
    if len(sys.argv) <= 1:
        return False

    if sys.argv[1].startswith('build'):
        return True

    if sys.argv[1].startswith('bdist'):
        return True

    if sys.argv[1].startswith('install'):
        return True


# Start to build c libs
# ------------------------------------------------
def test_compile(build_ext, name, suffix, code, libraries=None, include_dirs=None, library_dirs=None,
                 macros=None, extra_compile_preargs=None, extra_link_preargs=None):
    test_compile_dir = os.path.join(build_ext.build_temp, 'test_compile')
    if not os.path.exists(test_compile_dir):
        os.makedirs(test_compile_dir)

    source_file = os.path.join(test_compile_dir, '%s.%s' % (name, suffix))
    with open(source_file, 'w') as f:
        f.write(code)

    compiler = build_ext.compiler
    [object_file] = compiler.object_filenames([source_file])
    shared_object_file = compiler.shared_object_filename(
        name, output_dir=test_compile_dir)

    compiler.compile([source_file], extra_preargs=extra_compile_preargs,
                     include_dirs=include_dirs, macros=macros)
    compiler.link_shared_object(
        [object_file], shared_object_file, libraries=libraries, library_dirs=library_dirs,
        extra_preargs=extra_link_preargs)

    return shared_object_file

def get_cpp_flags(build_ext):
    last_err = None
    default_flags = ['-std=c++11', '-fPIC', '-Ofast', '-Wall', '-fopenmp', '-march=native']
    flags_to_try =  [default_flags, default_flags + ['-stdlib=libc++']]
    for cpp_flags in flags_to_try:
        try:
            test_compile(build_ext, 'test_cpp_flags', 'cc', extra_compile_preargs=cpp_flags,
                         code=textwrap.dedent('''\
                    #include <unordered_map>
                    void test() {
                    }
                    '''))

            return cpp_flags
        except (CompileError, LinkError):
            last_err = 'Unable to determine C++ compilation flags (see error above).'
        except Exception:
            last_err = 'Unable to determine C++ compilation flags.  ' \
                       'Last error:\n\n%s' % traceback.format_exc()

    raise DistutilsPlatformError(last_err)


def get_link_flags(build_ext):
    last_err = None
    libtool_flags = ['-Wl,-exported_symbols_list,samgraph.exp']
    ld_flags = ['-Wl,--version-script=samgraph.lds', '-fopenmp']
    flags_to_try = [ld_flags, libtool_flags]
    for link_flags in flags_to_try:
        try:
            test_compile(build_ext, 'test_link_flags', 'cc' , extra_link_preargs=link_flags,
                         code=textwrap.dedent('''\
                    void test() {
                    }
                    '''))

            return link_flags
        except (CompileError, LinkError):
            last_err = 'Unable to determine C++ link flags (see error above).'
        except Exception:
            last_err = 'Unable to determine C++ link flags.  ' \
                       'Last error:\n\n%s' % traceback.format_exc()

    raise DistutilsPlatformError(last_err)

def get_cuda_dirs(build_ext, cpp_flags):
    nvcc_compiler = None
    cuda_include_dirs = []
    cuda_lib_dirs = []

    cuda_home = os.environ.get('SAMGRAPH_CUDA_HOME')
    if cuda_home:
        nvcc_compiler = '%s/bin/nvcc' % cuda_home
        cuda_include_dirs += ['%s/include' % cuda_home]
        cuda_lib_dirs += ['%s/lib' % cuda_home, '%s/lib64' % cuda_home]

    cuda_include = os.environ.get('SAMGRAPH_CUDA_INCLUDE')
    if cuda_include:
        cuda_include_dirs += [cuda_include]

    cuda_lib = os.environ.get('SAMGRAPH_CUDA_LIB')
    if cuda_lib:
        cuda_lib_dirs += [cuda_lib]

    if not cuda_include_dirs and not cuda_lib_dirs:
        # default to /usr/local/cuda
        nvcc_compiler = '/usr/local/cuda/bin/nvcc'
        cuda_include_dirs += ['/usr/local/cuda/include']
        cuda_lib_dirs += ['/usr/local/cuda/lib', '/usr/local/cuda/lib64']

    try:
        test_compile(build_ext, 'test_cuda', 'cc', libraries=['cudart'], include_dirs=cuda_include_dirs,
                     library_dirs=cuda_lib_dirs, extra_compile_preargs=cpp_flags,
                     code=textwrap.dedent('''\
            #include <cuda_runtime.h>
            void test() {
                cudaSetDevice(0);
            }
            '''))
    except (CompileError, LinkError):
        raise DistutilsPlatformError(
            'CUDA library was not found (see error above).\n'
            'Please specify correct CUDA location with the SAMGRAPH_CUDA_HOME '
            'environment variable or combination of SAMGRAPH_CUDA_INCLUDE and '
            'SAMGRAPH_CUDA_LIB environment variables.\n\n'
            'SAMGRAPH_CUDA_HOME - path where CUDA include and lib directories can be found\n'
            'SAMGRAPH_CUDA_INCLUDE - path to CUDA include directory\n'
            'SAMGRAPH_CUDA_LIB - path to CUDA lib directory')

    return dict(NVCC_PATH=nvcc_compiler,
                CUDA_INCLUDE_DIRS=cuda_include_dirs, 
                CUDA_LIB_DIRS=cuda_lib_dirs)

def customize_compiler_for_nvcc(self, nvcc_path):
    """
    inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on.
    """

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', nvcc_path)
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            if isinstance(extra_postargs, dict):
                postargs = extra_postargs['NVCC_FLAGS']
            else:
                postargs = extra_postargs
        else:
            if isinstance(extra_postargs, dict):
                postargs = extra_postargs['GCC_FLAGS']
            else:
                postargs = extra_postargs

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile

def get_cuda_flags(build_ext):
    last_err = None
    default_flags = ['-std=c++11', '-arch=sm_35', '--ptxas-options=-v', '--compiler-options', "'-fPIC'"]
    flags_to_try =  [default_flags]
    for cuda_flags in flags_to_try:
        try:
            test_compile(build_ext, 'test_cuda_flags', 'cu', extra_compile_preargs=cuda_flags,
                         code=textwrap.dedent('''\
                    __global__ void test() {
                    }
                    '''))

            return cuda_flags
        except (CompileError, LinkError):
            last_err = 'Unable to determine Cuda compilation flags (see error above).'
        except Exception:
            last_err = 'Unable to determine Cuda compilation flags.  ' \
                       'Last error:\n\n%s' % traceback.format_exc()

    raise DistutilsPlatformError(last_err)

def get_common_options(build_ext):
    cpp_flags = get_cpp_flags(build_ext)
    link_flags = get_link_flags(build_ext)

    # set cuda support
    cuda_variables = get_cuda_dirs(build_ext, cpp_flags)
    customize_compiler_for_nvcc(build_ext.compiler, cuda_variables['NVCC_PATH'])
    cuda_flags = get_cuda_flags(build_ext)

    MACROS = []
    INCLUDES = []
    SOURCES = ['samgraph/common/saikyo.cc']
    
    COMPILE_FLAGS = cpp_flags
    LINK_FLAGS = link_flags

    LIBRARY_DIRS = []
    LIBRARIES = []

    EXTRA_OBJECTS = []
    
    CUDA_VARIABLES = cuda_variables
    CUDA_FLAGS = cuda_flags

    return dict(MACROS=MACROS,
                INCLUDES=INCLUDES,
                SOURCES=SOURCES,
                COMPILE_FLAGS=COMPILE_FLAGS,
                LINK_FLAGS=LINK_FLAGS,
                LIBRARY_DIRS=LIBRARY_DIRS,
                LIBRARIES=LIBRARIES,
                EXTRA_OBJECTS=EXTRA_OBJECTS,
                CUDA_VARIABLES=CUDA_VARIABLES,
                CUDA_FLAGS=CUDA_FLAGS)

def build_sampler(build_ext, options):
    options['INCLUDES'] += options['CUDA_VARIABLES']['CUDA_INCLUDE_DIRS']
    options['LIBRARY_DIRS'] += options['CUDA_VARIABLES']['CUDA_LIB_DIRS']
    options['LIBRARIES'] += ['cudart']

    sampler_lib.define_macros = options['MACROS']
    sampler_lib.include_dirs = options['INCLUDES']
    sampler_lib.sources = [
        'samgraph/common/saikyo.cc',
        'samgraph/sampler/test.cc',
        'samgraph/sampler/test_cuda.cu']
    sampler_lib.extra_compile_args = dict(
        GCC_FLAGS=options['COMPILE_FLAGS'],
        NVCC_FLAGS=options['CUDA_FLAGS']
    )
    sampler_lib.extra_link_args = options['LINK_FLAGS']
    sampler_lib.extra_objects = options['EXTRA_OBJECTS']
    sampler_lib.library_dirs = options['LIBRARY_DIRS']

    sampler_lib.libraries = []
    build_ext.build_extension(sampler_lib)

# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        options = get_common_options(self)
        try:
            build_sampler(self, options)
        except:
            raise DistutilsSetupError('An ERROR occured while building the server module.\n\n'
                                      '%s' % traceback.format_exc())

# Where the magic happens:
extensions_to_build = [sampler_lib]

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
    ext_modules=extensions_to_build,
    # $ setup.py publish support.
    cmdclass={
        'build_ext': custom_build_ext
    },
    # cffi is required for PyTorch
    # If cffi is specified in setup_requires, it will need libffi to be installed on the machine,
    # which is undesirable.  Luckily, `install` action will install cffi before executing build,
    # so it's only necessary for `build*` or `bdist*` actions.
    setup_requires=[]
)
