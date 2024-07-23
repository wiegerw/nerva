from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

import os
import sys

__version__ = "0.3"


define_macros = [('VERSION_INFO', __version__)]
include_dirs = ['include']
extra_compile_args = ['-DFMT_HEADER_ONLY', '-DNERVA_TIMER', '-DNERVA_USE_MKL']
extra_link_args = []

# set up Eigen
EIGENROOT = os.getenv('EIGENROOT')
define_macros += [('EIGEN_USE_MKL_ALL', 1)]
if not EIGENROOT:
    if not sys.platform.startswith("win"):
        EIGENROOT = '/usr/include/eigen3'
    else:    
        raise RuntimeError('environment variable EIGENROOT is not set')
include_dirs += [f'{EIGENROOT}']

# set up MKL
#
# N.B. On Windows the file libiomp5md.dll must be installed. It should be available in
# %ONEAPI_ROOT%\compiler\latest\windows\redist\intel64_win\compiler\libiomp5md.dll
# The easiest way to install it is to copy it to the directory C:\Windows\System32.
if sys.platform.startswith("win"):
    ONEAPI_ROOT = os.getenv('ONEAPI_ROOT')
    if not ONEAPI_ROOT:
        raise RuntimeError('environment variable ONEAPI_ROOT is not set')
    MKLROOT = f'{ONEAPI_ROOT}/mkl/latest'
    include_dirs += [f'{MKLROOT}/include']
    extra_compile_args += ['-DMKL_ILP64']
    extra_link_args += [f'{MKLROOT}/lib/intel64/mkl_intel_ilp64.lib',
                        f'{MKLROOT}/lib/intel64/mkl_intel_thread.lib',
                        f'{MKLROOT}/lib/intel64/mkl_core.lib',
                        f'{ONEAPI_ROOT}/compiler/latest/windows/compiler/lib/intel64_win/libiomp5md.lib'
                       ]
else:
    MKLROOT = os.getenv('MKLROOT')
    if not MKLROOT:
        if os.path.exists('/usr/include/mkl'):
            MKL_INCLUDE_DIR = '/usr/include/mkl'
            MKL_LIB_DIR = '/usr/lib/x86_64-linux-gnu'
        else:
            raise RuntimeError('environment variable MKLROOT is not set, and MKL is not found in the default location')
    else:
        MKL_INCLUDE_DIR = f'{MKLROOT}/include'
        MKL_LIB_DIR = f'{MKLROOT}/lib/intel64'

    include_dirs += [MKL_INCLUDE_DIR]
    extra_compile_args += ['-march=native', '-DMKL_ILP64', '-m64', '-fopenmp']
    extra_link_args += ['-Wl,--start-group',
                        f'{MKL_LIB_DIR}/libmkl_intel_ilp64.a',
                        f'{MKL_LIB_DIR}/libmkl_intel_thread.a',
                        f'{MKL_LIB_DIR}/libmkl_core.a',
                        '-Wl,--end-group',
                        '-liomp5',
                        '-lpthread',
                        '-lm',
                        '-ldl'
                       ]

ext_modules = [
    Pybind11Extension('nervalibcolwise',
        [ "src/logger.cpp", "src/python-bindings.cpp", "src/utilities.cpp" ],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=include_dirs,
        cxx_std=20
    )
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
    package_dir = {'': 'python'},
    packages=['nervacolwise']
)

