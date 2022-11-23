# Nerva
The Nerva library is a C++ library for neural networks. It includes
python bindings. Currently only multilayer perceptrons are supported with
dense or sparse layers. An example can be found in `python/cifar10.py`.

## Requirements
A C++17 compiler.

Compilation has been tested successfully with g++-11.3. Note that **the MKL library
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
It is expected that the location of the Intel MKL library is set
in the environment variable `MKLROOT`. The path to the Eigen include
files is currently hard coded as `/usr/include/eigen3`.
The following packages need to be installed:
```
libeigen3-dev
pybind11-dev
python3-pybind11
```
The python package can for example be installed using
```
pip3 install . --user
```

### Windows build
The Windows build has not been tested yet.