#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase

import numpy as np

import mlps.nerva_jax.matrix_operations as jnp_
import mlps.nerva_numpy.matrix_operations as np_
import mlps.nerva_sympy.matrix_operations as sympy_
import mlps.nerva_tensorflow.matrix_operations as tf_
import mlps.nerva_torch.matrix_operations as torch_
from mlps.tests.utilities import check_arrays_equal, check_numbers_equal, to_jax, to_numpy, to_sympy, to_tensorflow, \
    to_torch


class TestMatrixOperations(TestCase):
    def make_variables(self):
        m = 2
        n = 3

        def f(x):
            return x * x + 3

        X = np.array([
            [1, 2, 3],
            [7, 3, 4]
        ], dtype=float)

        Y = np.array([
            [2, 6, 6],
            [1, 4, 1]
        ], dtype=float)

        S = np.array([
            [1, 2, 3],
            [7, 3, 4],
            [8, 9, 5]
        ], dtype=float)

        xc = np.array([
            [7],
            [7],
            [9]
        ], dtype=float)

        xr = np.array([
            [2, 1, 9]
        ], dtype=float)

        return m, n, f, X, Y, S, xc, xr

    def test_zeros(self):
        #  (1) zeros(m: int, n: int = 1) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x1 = sympy_.zeros(m, n)
        x2 = np_.zeros(m, n)
        x3 = tf_.zeros(m, n)
        x4 = torch_.zeros(m, n)
        x5 = jnp_.zeros(m, n)
        check_arrays_equal(self, 'zeros', [x1, x2, x3, x4, x5])

    def test_ones(self):
        #  (2) ones(m: int, n: int = 1) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x1 = sympy_.ones(m, n)
        x2 = np_.ones(m, n)
        x3 = tf_.ones(m, n)
        x4 = torch_.ones(m, n)
        x5 = jnp_.ones(m, n)
        check_arrays_equal(self, 'ones', [x1, x2, x3, x4, x5])

    def test_identity(self):
        #  (3) identity(n: int) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x1 = sympy_.identity(n)
        x2 = np_.identity(n)
        x3 = tf_.identity(n)
        x4 = torch_.identity(n)
        x5 = jnp_.identity(n)
        check_arrays_equal(self, 'identity', [x1, x2, x3, x4, x5])

    def test_product(self):
        #  (4) product(X: Matrix, Y: Matrix) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x, y = X, S
        x1 = sympy_.product(to_sympy(x), to_sympy(y))
        x2 = np_.product(x, y)
        x3 = tf_.product(to_tensorflow(x), to_tensorflow(y))
        x4 = torch_.product(to_torch(x), to_torch(y))
        x5 = jnp_.product(to_jax(x), to_jax(y))
        check_arrays_equal(self, 'product', [x1, x2, x3, x4, x5])

    def test_hadamard(self):
        # (5) hadamard(X: Matrix, Y: Matrix) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x, y = X, Y
        x1 = sympy_.hadamard(to_sympy(x), to_sympy(y))
        x2 = np_.hadamard(x, y)
        x3 = tf_.hadamard(to_tensorflow(x), to_tensorflow(y))
        x4 = torch_.hadamard(to_torch(x), to_torch(y))
        x5 = jnp_.hadamard(to_jax(x), to_jax(y))
        check_arrays_equal(self, 'hadamard', [x1, x2, x3, x4, x5])

    def test_diag(self):
        # (6) diag(X: Matrix) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x = S
        x1 = sympy_.diag(to_sympy(x))
        x1 = to_numpy(x1).squeeze()  # Flatten the SymPy result
        x2 = np_.diag(x)
        x3 = tf_.diag(to_tensorflow(x))
        x4 = torch_.diag(to_torch(x))
        x5 = jnp_.diag(to_jax(x))
        check_arrays_equal(self, 'diag', [x1, x2, x3, x4, x5])

    def test_Diag(self):
        # (7) Diag(x: Matrix) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        for x in [xc, xr]:
            x1 = sympy_.Diag(to_sympy(x))
            x2 = np_.Diag(x)
            x3 = tf_.Diag(to_tensorflow(x))
            x4 = torch_.Diag(to_torch(x))
            x5 = jnp_.Diag(to_jax(x))
            check_arrays_equal(self, 'Diag', [x1, x2, x3, x4, x5])

    def test_elements_sum(self):
        # (8) elements_sum(X: Matrix):
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x = X
        x1 = sympy_.elements_sum(to_sympy(x))
        x2 = np_.elements_sum(x)
        x3 = tf_.elements_sum(to_tensorflow(x))
        x4 = torch_.elements_sum(to_torch(x))
        x5 = jnp_.elements_sum(to_jax(x))
        check_numbers_equal(self, 'elements_sum', [x1, x2, x3, x4, x5])

    def test_column_repeat(self):
        # (9) column_repeat(x: Matrix, n: int) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x = xc
        x1 = sympy_.column_repeat(to_sympy(x), n)
        x2 = np_.column_repeat(x, n)
        x3 = tf_.column_repeat(to_tensorflow(x), n)
        x4 = torch_.column_repeat(to_torch(x), n)
        x5 = jnp_.column_repeat(to_jax(x), n)
        check_arrays_equal(self, 'column_repeat', [x1, x2, x3, x4, x5])

    def test_row_repeat(self):
        # (10) row_repeat(xr: Matrix, m: int) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x = xr
        x1 = sympy_.row_repeat(to_sympy(x), n)
        x2 = np_.row_repeat(x, n)
        x3 = tf_.row_repeat(to_tensorflow(x), n)
        x4 = torch_.row_repeat(to_torch(x), n)
        x5 = jnp_.row_repeat(to_jax(x), n)
        check_arrays_equal(self, 'row_repeat', [x1, x2, x3, x4, x5])

    def test_columns_sum(self):
        # (11) columns_sum(X: Matrix) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x = X
        x1 = sympy_.columns_sum(to_sympy(x))
        x2 = np_.columns_sum(x)
        x3 = tf_.columns_sum(to_tensorflow(x))
        x4 = torch_.columns_sum(to_torch(x))
        x5 = jnp_.columns_sum(to_jax(x))
        check_arrays_equal(self, 'columns_sum', [x1, x2, x3, x4, x5])

    def test_rows_sum(self):
        # (12) rows_sum(X: Matrix) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x = X
        x1 = sympy_.rows_sum(to_sympy(x))
        x1 = to_numpy(x1).squeeze()  # Flatten the SymPy result
        x2 = np_.rows_sum(x)
        x3 = tf_.rows_sum(to_tensorflow(x))
        x4 = torch_.rows_sum(to_torch(x))
        x5 = jnp_.rows_sum(to_jax(x))
        check_arrays_equal(self, 'rows_sum', [x1, x2, x3, x4, x5])

    def test_columns_max(self):
        # (13) columns_max(X: Matrix) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x = X
        x1 = sympy_.columns_max(to_sympy(x))
        x2 = np_.columns_max(x)
        x3 = tf_.columns_max(to_tensorflow(x))
        x4 = torch_.columns_max(to_torch(x))
        x5 = jnp_.columns_max(to_jax(x))
        check_arrays_equal(self, 'columns_max', [x1, x2, x3, x4, x5])

    def test_rows_max(self):
        # (14) rows_max(X: Matrix) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x = X
        x1 = sympy_.rows_max(to_sympy(x))
        x1 = to_numpy(x1).squeeze()  # Flatten the SymPy result
        x2 = np_.rows_max(x)
        x3 = tf_.rows_max(to_tensorflow(x))
        x4 = torch_.rows_max(to_torch(x))
        x5 = jnp_.rows_max(to_jax(x))
        check_arrays_equal(self, 'rows_max', [x1, x2, x3, x4, x5])

    def test_columns_mean(self):
        # (15) columns_mean(X: Matrix) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x = X
        x1 = sympy_.columns_mean(to_sympy(x))
        x2 = np_.columns_mean(x)
        x3 = tf_.columns_mean(to_tensorflow(x))
        x4 = torch_.columns_mean(to_torch(x))
        x5 = jnp_.columns_mean(to_jax(x))
        check_arrays_equal(self, 'columns_mean', [x1, x2, x3, x4, x5])

    def test_rows_mean(self):
        # (16) rows_mean(X: Matrix) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x = X
        x1 = sympy_.rows_mean(to_sympy(x))
        x1 = to_numpy(x1).squeeze()  # Flatten the SymPy result
        x2 = np_.rows_mean(x)
        x3 = tf_.rows_mean(to_tensorflow(x))
        x4 = torch_.rows_mean(to_torch(x))
        x5 = jnp_.rows_mean(to_jax(x))
        check_arrays_equal(self, 'rows_mean', [x1, x2, x3, x4, x5])

    def test_apply(self):
        # (17) apply(f, X: Matrix) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x = X
        x1 = sympy_.apply(f, to_sympy(x))
        x2 = np_.apply(f, x)
        x3 = tf_.apply(f, to_tensorflow(x))
        x4 = torch_.apply(f, to_torch(x))
        x5 = jnp_.apply(f, to_jax(x))
        check_arrays_equal(self, 'apply', [x1, x2, x3, x4, x5])

    def test_exp(self):
        # (18) exp(X: Matrix) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x = X
        x1 = sympy_.exp(to_sympy(x))
        x2 = np_.exp(x)
        x3 = tf_.exp(to_tensorflow(x))
        x4 = torch_.exp(to_torch(x))
        x5 = jnp_.exp(to_jax(x))
        check_arrays_equal(self, 'exp', [x1, x2, x3, x4, x5])

    def test_log(self):
        # (19) log(X: Matrix) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x = X
        x1 = sympy_.log(to_sympy(x))
        x2 = np_.log(x)
        x3 = tf_.log(to_tensorflow(x))
        x4 = torch_.log(to_torch(x))
        x5 = jnp_.log(to_jax(x))
        check_arrays_equal(self, 'log', [x1, x2, x3, x4, x5])

    def test_inverse(self):
        # (20) inverse(X: Matrix) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x = X
        x1 = sympy_.inverse(to_sympy(x))
        x2 = np_.inverse(x)
        x3 = tf_.inverse(to_tensorflow(x))
        x4 = torch_.inverse(to_torch(x))
        x5 = jnp_.inverse(to_jax(x))
        check_arrays_equal(self, 'inverse', [x1, x2, x3, x4, x5])

    def test_square(self):
        # (21) square(X: Matrix) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x = X
        x1 = sympy_.square(to_sympy(x))
        x2 = np_.square(x)
        x3 = tf_.square(to_tensorflow(x))
        x4 = torch_.square(to_torch(x))
        x5 = jnp_.square(to_jax(x))
        check_arrays_equal(self, 'square', [x1, x2, x3, x4, x5])

    def test_sqrt(self):
        # (22) sqrt(X: Matrix) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x = X
        x1 = sympy_.sqrt(to_sympy(x))
        x2 = np_.sqrt(x)
        x3 = tf_.sqrt(to_tensorflow(x))
        x4 = torch_.sqrt(to_torch(x))
        x5 = jnp_.sqrt(to_jax(x))
        check_arrays_equal(self, 'sqrt', [x1, x2, x3, x4, x5])

    def test_power_minus_half(self):
        # (23) power_minus_half(X: Matrix) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x = X
        x1 = sympy_.power_minus_half(to_sympy(x))
        x2 = np_.power_minus_half(x)
        x3 = tf_.power_minus_half(to_tensorflow(x))
        x4 = torch_.power_minus_half(to_torch(x))
        x5 = jnp_.power_minus_half(to_jax(x))
        check_arrays_equal(self, 'power_minus_half', [x1, x2, x3, x4, x5])


    def test_log_sigmoid(self):
        # (24) log_sigmoid(X: Matrix) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x = X
        x1 = sympy_.log_sigmoid(to_sympy(x))
        x2 = np_.log_sigmoid(x)
        x3 = tf_.log_sigmoid(to_tensorflow(x))
        x4 = torch_.log_sigmoid(to_torch(x))
        x5 = jnp_.log_sigmoid(to_jax(x))
        check_arrays_equal(self, 'log_sigmoid', [x1, x2, x3, x4, x5])


if __name__ == '__main__':
    import unittest
    unittest.main()
