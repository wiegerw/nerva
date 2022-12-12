# Nerva
The Nerva library is a C++ library for neural networks. It includes
python bindings. Currently only multilayer perceptrons are supported with
dense or sparse layers. An example can be found in `python/cifar10.py`.

## Requirements
A C++17 compiler and an Intel processor (due to the dependency on Intel MKL).

Compilation has been tested successfully with g++-11.3 and Visual Studio 2019. Note that **the MKL library
must be linked statically**. It turned out that shared linking causes wrong results
in one of the MKL sparse matrix multiplication routines.

Compilation with
clang-14 was tried, but unfortunately the compiler + linker flags mentioned
on the web page https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html
do not seem to work.

## Dependencies
Nerva uses the following third-party libraries.

* doctest (https://github.com/onqtam/doctest, already included in the repository)
* fmt (https://github.com/fmtlib/fmt, already included in the repository)
* Eigen (https://eigen.tuxfamily.org/)
* pybind11 (https://github.com/pybind/pybind11)
* Intel MKL (https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html)

## Build
The following build systems are supported
* CMake 3.16+
* B2 (https://www.bfgroup.xyz/b2/)

### Ubuntu build
The Ubuntu build has been tested for both Ubuntu Focal and Ubuntu Jammy. There are docker
files available in the `docker` subdirectory that contain the exact instructions that
were used for the build.

The following packages need to be installed:
```
libeigen3-dev
libmkl-dev
pybind11-dev
python3-pybind11
```
The nerva package can for example be installed using
```
pip3 install .
```

It's also possible to install the Eigen and MKL libraries manually.
In that case the location of the Intel MKL library must be set
in the environment variable `MKLROOT`, and the location of the Eigen
library must be set in the environment variable `EIGENROOT`. Also
the `LD_LIBRARY_PATH` must be extended, for example by
setting the following variables in `.bashrc`: 
```
export MKLROOT=/opt/intel/oneapi/mkl/2022.2.1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${MKLROOT}/lib/intel64
export EIGENROOT=/path/to/eigen-3.4.0
```

The number of cores that is used can be controlled using environment variables:
```
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4
```

### Windows build
The Windows build is still experimental. 

The location of the oneAPI library must be set in the environment variable `ONEAPI_ROOT`,
and the location of the Eigen library must be set in the environment variable `EIGENROOT`.
Note that the default installation of Intel MKL will set `ONEAPI_ROOT` automatically.
Also the file `libiomp5md.dll` must be installed. It can be found
in `%ONEAPI_ROOT%\compiler\latest\windows\redist\intel64_win\compiler\libiomp5md.dll`.
The easiest way to install it is to copy it manually to the directory `C:\Windows\System32`.

To get optimal performance, it may be needed to add compiler flags like `/arch:AVX2`
manually to the `extra_compile_args` list in the file `setup.py`. Unfortunately on Windows
there seems to be no standard way to automatically select the correct flags. The nerva package can
for example be installed using
```
pip3 install .
```
Installing it with the flag `--user` should also work, but then it seems
necessary to add the local python install directory to the Windows PATH
manually.