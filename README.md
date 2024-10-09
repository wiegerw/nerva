![Nerva Logo](images/nerva-logo.png)
# Nerva

> **Note:**  
> N.B. The original `main` branch of this repository has been renamed to `main_old`. This branch contains the historical development of the code base, including the sparse neural network experiments that are reported in [1]. All important functionality has been separated into a number of independent repositories. Moreover, a substantial amount of documentation has been added to the C++ libraries. Similar documentation for the Python libraries is on the way.

The Nerva framework consists of a collection of C++ and Python libraries for neural networks. Originally the library was developed for experimenting with truly sparse neural networks in C++. In the meantime, several Python implementations have been added. The following repositories are available:

| Repository                                  | Description                                              |
|---------------------------------------------|----------------------------------------------------------|
| https://github.com/wiegerw/nerva-rowwise    | A C++ implementation with data in row-wise layout        |
| https://github.com/wiegerw/nerva-colwise    | A C++ implementation with data in column-wise layout     |
| https://github.com/wiegerw/nerva-jax        | A Python implementation using JAX data structures        |
| https://github.com/wiegerw/nerva-numpy      | A Python implementation using NumPy data structures      |
| https://github.com/wiegerw/nerva-tensorflow | A Python implementation using TensorFlow data structures |
| https://github.com/wiegerw/nerva-torch      | A Python implementation using PyTorch data structures    |
| https://github.com/wiegerw/nerva-sympy      | A symbolic Python implementation used for validation     |

## Features
The Nerva libraries have the following features:
* They support common layers, loss functions and activation functions.
* They support mini-batches, and all equations (including backpropagation!) are given in matrix form.
* Precise mathematical specifications of the equations are available in this [specification document](https://wiegerw.github.io/nerva-rowwise/pdf/nerva-library-specifications.pdf). 

The `nerva-colwise` and `nerva-rowwise` libraries have the following additional features:
* They support truly sparse layers. The weight matrices of these layers are stored using a sparse matrix representation (CSR).
* They include Python bindings.

## Limitations
* Only multilayer perceptrons are supported.
* Only the CPU is supported.

## References
The following papers about Nerva are available:

[1] *Nerva: a Truly Sparse Implementation of Neural Networks*,  https://arxiv.org/abs/2407.17437. It introduces the library, and describes a number of static sparse training experiments.

[2] *Batch Matrix-form Equations and Implementation
of Multilayer Perceptrons*, https://arxiv.org/abs/TODO. It describes the implementation of the Nerva libraries in great detail.

## Contact
If you have questions, or if you would like to contribute to the Nerva libraries, you can email Wieger Wesselink (j.w.wesselink@tue.nl).
