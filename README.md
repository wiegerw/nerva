![Nerva Logo](images/nerva-logo.png)

# Nerva

[![License: BSL-1.0](https://img.shields.io/badge/License-Boost%201.0-lightblue.svg)](LICENSE)

**Nerva** is a collection of modular C++ and Python libraries for building and experimenting with neural networks. Originally developed for research on **truly sparse neural networks**, the Nerva libraries have evolved into a set of well-documented, flexible components that emphasize mathematical clarity and transparent implementation.

> **Note:**  
> The original development history can be found on the [`main_old`](https://github.com/wiegerw/nerva/tree/main_old) branch. This includes all sparse training experiments reported in [1]. The functionality has since been split into independent repositories, each with focused documentation and clean interfaces.

---

## Repositories

The Nerva project is organized into a set of focused libraries, each targeting a specific language or data layout:

| Repository                                                      | Description                                                  |
|-----------------------------------------------------------------|--------------------------------------------------------------|
| [**nerva-rowwise**](https://github.com/wiegerw/nerva-rowwise)       | C++ backend with row-major layout (like PyTorch)             |
| [**nerva-colwise**](https://github.com/wiegerw/nerva-colwise)       | C++ backend with column-major layout (like MATLAB)           |
| [**nerva-jax**](https://github.com/wiegerw/nerva-jax)               | Python implementation using JAX                              |
| [**nerva-numpy**](https://github.com/wiegerw/nerva-numpy)           | Python implementation using NumPy                            |
| [**nerva-tensorflow**](https://github.com/wiegerw/nerva-tensorflow) | Python implementation using TensorFlow                       |
| [**nerva-torch**](https://github.com/wiegerw/nerva-torch)           | Python implementation using PyTorch                          |
| [**nerva-sympy**](https://github.com/wiegerw/nerva-sympy)           | Symbolic Python backend for specification and validation     |

---

## Features

The Nerva libraries share a common design:

- ✅ Support for common layers, activation functions, and loss functions  
- ✅ Matrix-form equations for all operations, including **explicit backpropagation**  
- ✅ Mini-batch training via matrix algebra  
- ✅ Symbolic validation of equations using SymPy
- ✅ Uniform implementations grounded in a small set of matrix primitives

**C++ Libraries Only** (`nerva-rowwise`, `nerva-colwise`):

- ✅ Native CPU performance using Intel MKL  
- ✅ Support for **truly sparse** layers (using CSR representation)  
- ✅ Python bindings available for easy integration

---

## Limitations

- ⚠️ Currently limited to **multilayer perceptrons** (MLPs)  
- ⚠️ **GPU support** is not yet available  
- ⚠️ No convolutional or transformer layers (planned)

---

## References

1. **Nerva: a Truly Sparse Implementation of Neural Networks**

   [arXiv:2407.17437](https://arxiv.org/abs/2407.17437)
   Introduces the library and reports sparse training experiments.

2. **Batch Matrix-form Equations and Implementation of Multilayer Perceptrons**

   [arXiv:2511.11918](https://arxiv.org/abs/2511.11918)
   Includes mathematical specifications and derivations.

---

## Contact

Questions or contributions are welcome!  
Contact: **Wieger Wesselink** (j.w.wesselink@tue.nl)

---

