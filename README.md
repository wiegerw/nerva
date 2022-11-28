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
It is expected that the location of the Intel MKL library is set
in the environment variable `MKLROOT`, and the location of the Eigen
library is set in the environment variable `EIGENROOT`. If the latter
variable is not set, the Eigen include files are assumed to be in
`/usr/include/eigen3`.
The following packages need to be installed:
```
libeigen3-dev
pybind11-dev
python3-pybind11
```
The nerva package can for example be installed using
```
pip3 install . --user
```

### Windows build
The Windows build is still experimental. It is expected that the location of the
oneAPI library is set in the environment variable `ONEAPI_ROOT`, and the
location of the Eigen library is set in the environment variable `EIGENROOT`.
Note that the default installation of Intel MKL will set `ONEAPI_ROOT` automatically.
On Windows the file `libiomp5md.dll` must be installed. It can be found
in `%ONEAPI_ROOT%\compiler\latest\windows\redist\intel64_win\compiler\libiomp5md.dll`.
The easiest way to install it is to copy it manually to the directory `C:\Windows\System32`.
The performance doesn't seem optimal yet. Setting a compiler flag like `/arch:AVX2`
may improve the performance, but there seems to be no standard way to automatically
choose the right combination of flags.
The nerva package can for example be installed using
```
pip3 install .
```
Installing it with the flag `--user` should also work, but then it seems
necessary to add the local python install directory to the Windows PATH
manually.