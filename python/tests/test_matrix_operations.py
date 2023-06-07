#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from typing import Union
from unittest import TestCase

import numpy as np
import sympy as sp
import torch
import tensorflow as tf

# Set the environment variable to use CPU only
tf.config.set_visible_devices([], 'GPU')


import symbolic.matrix_operations_numpy as np_
import symbolic.matrix_operations_tensorflow as tf_
import symbolic.matrix_operations_torch as torch_
import symbolic.matrix_operations as sympy_


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


class TestMatrixOperationsNumPy(TestCase):

    def check_arrays_equal(self, operation, x1, x2, x3, x4):
        print(f'--- {operation} ---')
        x1 = to_numpy(x1)
        x2 = to_numpy(x2)
        x3 = to_numpy(x3)
        x4 = to_numpy(x4)
        print(x1)
        print(x2)
        print(x3)
        print(x4)
        self.assertTrue(np.allclose(x1, x2))
        self.assertTrue(np.allclose(x1, x3))
        self.assertTrue(np.allclose(x1, x4))
        # self.assertTrue(np.allclose(x1, x4, atol=1e-2))

    def test_operations(self):
        m = 2
        n = 3

        def f(x):
            return x * x + 3

        X = np.array([
            [1, 2, 3],
            [7, 3, 4]
        ])

        Y = np.array([
            [2, 6, 6],
            [1, 4, 1]
        ])

        S = np.array([
            [1, 2, 3],
            [7, 3, 4],
            [8, 9, 5]
        ])

        x = np.array([
            [2],
            [5],
            [4]
        ])

        xc = np.array([
            [7],
            [7],
            [9]
        ])

        xr = np.array(
            [2, 1, 9]
        )

        #  (1) zeros(m: int, n: int = 1) -> Matrix:
        x1 = sympy_.zeros(m, n)
        x2 = np_.zeros(m, n)
        x3 = tf_.zeros(m, n)
        x4 = torch_.zeros(m, n)
        self.check_arrays_equal('zeros', x1, x2, x3, x4)

        #  (2) ones(m: int, n: int = 1) -> Matrix:
        x1 = sympy_.ones(m, n)
        x2 = np_.ones(m, n)
        x3 = tf_.ones(m, n)
        x4 = torch_.ones(m, n)
        self.check_arrays_equal('ones', x1, x2, x3, x4)

        #  (3) identity(n: int) -> Matrix:
        x1 = sympy_.identity(n)
        x2 = np_.identity(n)
        x3 = tf_.identity(n)
        x4 = torch_.identity(n)
        self.check_arrays_equal('identity', x1, x2, x3, x4)

        #  (4) product(X: Matrix, Y: Matrix) -> Matrix:
        x1 = sympy_.product(to_sympy(X), to_sympy(S))
        x2 = np_.product(X, S)
        x3 = tf_.product(to_tensorflow(X), to_tensorflow(S))
        x4 = torch_.product(to_torch(X), to_torch(S))
        self.check_arrays_equal('product', x1, x2, x3, x4)

        # (5) hadamard(X: Matrix, Y: Matrix) -> Matrix:
        x1 = sympy_.hadamard(to_sympy(X), to_sympy(Y))
        x2 = np_.hadamard(X, Y)
        x3 = tf_.hadamard(to_tensorflow(X), to_tensorflow(Y))
        x4 = torch_.hadamard(to_torch(X), to_torch(Y))
        self.check_arrays_equal('hadamard', x1, x2, x3, x4)

        # (6) diag(X: Matrix) -> Matrix:
        x1 = sympy_.diag(to_sympy(S))
        x2 = np_.diag(S)
        x3 = tf_.diag(to_tensorflow(S))
        x4 = torch_.diag(to_torch(S))
        self.check_arrays_equal('diag', x1, x2, x3, x4)

        # (7) Diag(x: Matrix) -> Matrix:
        # (8) elements_sum(X: Matrix):
        # (9) column_repeat(xc: Matrix, n: int) -> Matrix:
        # (10) row_repeat(xr: Matrix, m: int) -> Matrix:
        # (11) columns_sum(X: Matrix) -> Matrix:
        # (12) rows_sum(X: Matrix) -> Matrix:
        # (13) columns_max(X: Matrix) -> Matrix:
        # (14) rows_max(X: Matrix) -> Matrix:
        # (15) columns_mean(X: Matrix) -> Matrix:
        # (16) rows_mean(X: Matrix) -> Matrix:
        # (17) apply(f, X: Matrix) -> Matrix:
        # (18) exp(X: Matrix) -> Matrix:
        # (19) log(X: Matrix) -> Matrix:
        # (20) inverse(X: Matrix) -> Matrix:
        # (21) square(X: Matrix) -> Matrix:
        # (22) sqrt(X: Matrix) -> Matrix:
        # (23) power_minus_half(X: Matrix) -> Matrix:


if __name__ == '__main__':
    import unittest
    unittest.main()
