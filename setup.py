from glob import glob
from setuptools import setup

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension
from pybind11 import get_cmake_dir

import os
import sys

__version__ = "0.12"

MKLROOT = os.getenv('MKLROOT')

define_macros = [('VERSION_INFO', __version__)]
extra_compile_args = ['-DFMT_HEADER_ONLY']
extra_link_args = ['-ltbb'] if not sys.platform.startswith("win") else []
include_dirs = ['include']

if not MKLROOT:
    raise RuntimeError('environment variable MKLROOT is not set')

define_macros += [('EIGEN_USE_MKL_ALL', 1)]
include_dirs += [f'{MKLROOT}/include']

if sys.platform.startswith("win"):
    extra_compile_args += ['-DMKL_ILP64']
    extra_link_args += ['mkl_intel_ilp64.lib',
                        'mkl_intel_thread.lib'
                        'mkl_core.lib',
                        'libiomp5md.lib'
                       ]
else:
    include_dirs += ['/usr/include/eigen3']  # TODO: find a portable solution
    extra_compile_args += ['-march=native', '-DMKL_ILP64', '-m64']
    extra_link_args += ['-Wl,--start-group',
                        f'{MKLROOT}/lib/intel64/libmkl_intel_ilp64.a',
                        f'{MKLROOT}/lib/intel64/libmkl_intel_thread.a',
                        f'{MKLROOT}/lib/intel64/libmkl_core.a',
                        '-Wl,--end-group',
                        '-liomp5',
                        '-lpthread',
                        '-lm',
                        '-ldl'
                       ]


ext_modules = [
    Pybind11Extension("nervalib",
        [ "src/logger.cpp", "src/python-bindings.cpp", "src/utilities.cpp" ],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=include_dirs,
        cxx_std=17
    ),
]

setup(
    name="nerva",
    version=__version__,
    author="Wieger Wesselink",
    author_email="j.w.wesselink@tue.nl",
    description="C++ library for Neural Networks",
    long_description="",
    ext_modules=ext_modules,
    zip_safe=False,
    packages=['nerva']
)
