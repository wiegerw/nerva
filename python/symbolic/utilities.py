# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import random
import time
from typing import Union

import numpy as np
import sympy as sp
import tensorflow as tf
import torch
from sympy import Matrix


def matrix(name: str, rows: int, columns: int) -> Matrix:
    return Matrix(sp.symarray(name, (rows, columns), real=True))


def equal_matrices(A: Matrix, B: Matrix, simplify_arguments=False) -> bool:
    m, n = A.shape
    if simplify_arguments:
        A = sp.simplify(A)
        B = sp.simplify(B)
    return A.shape == B.shape and sp.simplify(A - B) == sp.zeros(m, n)


def instantiate(X: sp.Matrix, low=0, high=10) -> sp.Matrix:
    X0 = sp.Matrix(np.random.randint(low, high, X.shape))
    return X0


def instantiate_one_hot_colwise(X: sp.Matrix) -> sp.Matrix:
    m, n = X.shape
    X0 = sp.zeros(m, n)
    for j in range(n):
        i = random.randrange(0, m)
        X0[i, j] = 1

    return X0


def instantiate_one_hot_rowwise(X: sp.Matrix) -> sp.Matrix:
    m, n = X.shape
    X0 = sp.zeros(m, n)
    for i in range(m):
        j = random.randrange(0, n)
        X0[i, j] = 1

    return X0


def to_numpy(x: Union[sp.Matrix, np.ndarray, torch.Tensor, tf.Tensor]) -> np.ndarray:
    if isinstance(x, sp.Matrix):
        return np.array(x.tolist(), dtype=np.float64)
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, tf.Tensor):
        return x.numpy()
    else:
        raise ValueError("Unsupported input type. Input must be one of sp.Matrix, np.ndarray, torch.Tensor, or tf.Tensor.")


def to_sympy(X: np.ndarray) -> sp.Matrix:
    return sp.Matrix(X)


def to_torch(X: np.ndarray) -> torch.Tensor:
    return torch.Tensor(X)


def to_tensorflow(X: np.ndarray) -> tf.Tensor:
    return tf.convert_to_tensor(X)


def squared_error(X: Matrix):
    m, n = X.shape

    def f(x: Matrix) -> float:
        return sp.sqrt(sum(xi * xi for xi in x))

    return sum(f(X.col(j)) for j in range(n))


class StopWatch(object):
    def __init__(self):
        self.start = time.perf_counter()

    def seconds(self):
        end = time.perf_counter()
        return end - self.start

    def reset(self):
        self.start = time.perf_counter()


def pp(name: str, x: Union[torch.Tensor, np.ndarray, tf.Tensor]):
    if isinstance(x, (np.ndarray, tf.Tensor)):
        x = torch.Tensor(x)
    if len(x.shape) == 1:
        print(f'{name} ({x.shape[0]})\n{x.data}')
    else:
        print(f'{name} ({x.shape[0]}x{x.shape[1]})\n{x.data}')
