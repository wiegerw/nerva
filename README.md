# Nerva
The Nerva library is a C++ library for neural networks. It includes
python bindings. Currently only multilayer perceptrons are supported with
dense or sparse layers. An example can be found in `python/cifar10.py`.

## Requirements
A C++17 compiler and an Intel processor (due to the dependency on Intel MKL).

Compilation has been tested successfully with g++-11.3, g++-12.1 and Visual Studio 2019.

Compilation with clang-14 was tried, but unfortunately the compiler + linker flags mentioned
on the web page https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html
do not seem to work.

## Dependencies
Nerva uses the following third-party libraries.

* doctest (https://github.com/onqtam/doctest, already included in the repository)
* fmt (https://github.com/fmtlib/fmt, already included in the repository)
* lyra (https://github.com/bfgroup/Lyra, already included in the repository)
* Eigen (https://eigen.tuxfamily.org/)
* pybind11 (https://github.com/pybind/pybind11)
* Intel MKL (https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html)

## Python build
This section explains how to install the nerva python module on Ubuntu and on Windows.

### Ubuntu
The Ubuntu build has been tested for both Ubuntu Focal and Ubuntu Jammy. There are docker
files available in the `docker` subdirectory that contain the exact instructions that
were used for the build.

The following packages need to be installed:
```
libeigen3-dev
libmkl-dev
pybind11-dev
python3-pybind11
build-essential   # meta-packages that are necessary for compiling software
python3-pip       # for installing python packages using pip
```
The nerva package can then be installed using
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

### Windows
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
then be installed using
```
pip3 install .
```
Installing it with the flag `--user` should also work, but then it seems
necessary to add the local python install directory to the Windows PATH
manually.

## C++ build
The following build systems are supported
* CMake 3.16+
* B2 (https://www.bfgroup.xyz/b2/)

Using CMake, the nerva c++ library can be built in the standard way. For example compiling
the nerva library on Ubuntu and running the tests can be done like this:
```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE
make -j4
ctest
```

### Intel MKL + CMake
N.B. The CMake support for the MKL library seems broken. There is no CMake support
available if the MKL library is installed using the package manager. Recent
versions provide CMake support, but there are some problems with it. First of all,
the Intel TBB library seems to be a requirement, which should not be the case.
Second, the CMake configuration doesn't set the proper flags for the g++ and
clang compilers. To fix this, the flags `-DMKL_ILP64 -m64 -march=native` are added
manually.

### Number type
By default, the nerva code uses 32-bit floats as the number type. It is possible to change this by
defining the symbol `NERVA_USE_DOUBLE`, in which case 64 bit doubles are used. The test
`tests/gradient_test.cpp` usually fails when the number type float is used, due to a lack of precision.

### The tool mlp
In the folder `tools` a command line program called `mlp` can be found that demonstrates the capabilities
of the nerva library. For example the following command can be used to train a neural network on a
simple dataset that is generated on the fly.
```
mlp --architecture=RRL --loss=squared-error --weights=xxx --hidden=64,64 --epochs=100 \
    --dataset=chessboard --learning-rate="constant(0.01)" --size=20000 --batch-size=10 \ 
    --normalize --threads=4 --no-shuffle --seed=1885661379 -v --algorithm=sgd
```
The output looks like this:
```
loading dataset chessboard
number of examples: 20000
number of features: 2
number of outputs: 2
epochs = 100
batch size = 1
shuffle = false
statistics = true
debug = false
algorithm = sgd
dataset = chessboard
dataset size = 20000
normalize data = true
learning rate scheduler = constant(0.01)
loss function = squared-error
architecture = RRL
sizes = [2, 64, 64, 2]
weights initialization = xxx
optimizer = gradient-descent
dropout = 0
sparsity = 0
seed = 1885661379
precision = 4
threads = 4
number type = float

epoch   0  loss:  0.5503  train accuracy:  0.4965  test accuracy:  0.4883  time:  0.0000s
epoch   1  loss:  0.2500  train accuracy:  0.5020  test accuracy:  0.5080  time:  0.0704s
epoch   2  loss:  0.2499  train accuracy:  0.5020  test accuracy:  0.5080  time:  0.0688s
epoch   3  loss:  0.2498  train accuracy:  0.5114  test accuracy:  0.5178  time:  0.0692s
epoch   4  loss:  0.2496  train accuracy:  0.5147  test accuracy:  0.5198  time:  0.0683s
epoch   5  loss:  0.2495  train accuracy:  0.5120  test accuracy:  0.5150  time:  0.0684s
epoch   6  loss:  0.2493  train accuracy:  0.5080  test accuracy:  0.5115  time:  0.0688s
epoch   7  loss:  0.2491  train accuracy:  0.5060  test accuracy:  0.5102  time:  0.0686s
epoch   8  loss:  0.2487  train accuracy:  0.5071  test accuracy:  0.5110  time:  0.0685s
epoch   9  loss:  0.2483  train accuracy:  0.5080  test accuracy:  0.5108  time:  0.0686s
epoch  10  loss:  0.2479  train accuracy:  0.5081  test accuracy:  0.5112  time:  0.0686s
```

Using `--dataset=cifar10` the `CIFAR10` dataset can be loaded. However, it will not be loaded
automatically. Instead, the location of the data is hardcoded as `../data/cifar-10-batches-bin`.