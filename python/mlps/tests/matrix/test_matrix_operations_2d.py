#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase
import numpy as np
import mlps.tests.matrix.matrix_operations_numpy as np_
import mlps.tests.matrix.matrix_operations_tensorflow as tf_
import mlps.tests.matrix.matrix_operations_torch as torch_
import mlps.tests.matrix.matrix_operations_sympy as sympy_
import mlps.tests.matrix.matrix_operations_jax as jnp_
from mlps.tests.test_utilities import to_numpy, to_sympy, to_torch, to_tensorflow, to_jax


class TestMatrixOperations(TestCase):

    def check_arrays_equal(self, operation, values):
        print(f'--- {operation} ---')
        values = [to_numpy(x) for x in values]
        for x in values:
            print(x)
        x0 = values[0]
        for x in values[1:]:
            self.assertTrue(np.allclose(x0, x, atol=1e-5))

    def check_numbers_equal(self, operation, values):
        print(f'--- {operation} ---')
        for x in values:
            print(x, x.__class__)
        x0 = values[0]
        for x in values[1:]:
            self.assertAlmostEqual(x0, x, delta=1e-5)

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
        self.check_arrays_equal('zeros', [x1, x2, x3, x4, x5])

    def test_ones(self):
        #  (2) ones(m: int, n: int = 1) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x1 = sympy_.ones(m, n)
        x2 = np_.ones(m, n)
        x3 = tf_.ones(m, n)
        x4 = torch_.ones(m, n)
        x5 = jnp_.ones(m, n)
        self.check_arrays_equal('ones', [x1, x2, x3, x4, x5])

    def test_identity(self):
        #  (3) identity(n: int) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x1 = sympy_.identity(n)
        x2 = np_.identity(n)
        x3 = tf_.identity(n)
        x4 = torch_.identity(n)
        x5 = jnp_.identity(n)
        self.check_arrays_equal('identity', [x1, x2, x3, x4, x5])

    def test_product(self):
        #  (4) product(X: Matrix, Y: Matrix) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x, y = X, S
        x1 = sympy_.product(to_sympy(x), to_sympy(y))
        x2 = np_.product(x, y)
        x3 = tf_.product(to_tensorflow(x), to_tensorflow(y))
        x4 = torch_.product(to_torch(x), to_torch(y))
        x5 = jnp_.product(to_jax(x), to_jax(y))
        self.check_arrays_equal('product', [x1, x2, x3, x4, x5])

    def test_hadamard(self):
        # (5) hadamard(X: Matrix, Y: Matrix) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x, y = X, Y
        x1 = sympy_.hadamard(to_sympy(x), to_sympy(y))
        x2 = np_.hadamard(x, y)
        x3 = tf_.hadamard(to_tensorflow(x), to_tensorflow(y))
        x4 = torch_.hadamard(to_torch(x), to_torch(y))
        x5 = jnp_.hadamard(to_jax(x), to_jax(y))
        self.check_arrays_equal('hadamard', [x1, x2, x3, x4, x5])

    def test_diag(self):
        # (6) diag(X: Matrix) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x = S
        x1 = sympy_.diag(to_sympy(x))
        x2 = np_.diag(x)
        x3 = tf_.diag(to_tensorflow(x))
        x4 = torch_.diag(to_torch(x))
        x5 = jnp_.diag(to_jax(x))
        self.check_arrays_equal('diag', [x1, x2, x3, x4, x5])

    def test_Diag(self):
        # (7) Diag(x: Matrix) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        for x in [xc, xr]:
            x1 = sympy_.Diag(to_sympy(x))
            x2 = np_.Diag(x)
            x3 = tf_.Diag(to_tensorflow(x))
            x4 = torch_.Diag(to_torch(x))
            x5 = jnp_.Diag(to_jax(x))
            self.check_arrays_equal('Diag', [x1, x2, x3, x4, x5])

    def test_elements_sum(self):
        # (8) elements_sum(X: Matrix):
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x = X
        x1 = sympy_.elements_sum(to_sympy(x))
        x2 = np_.elements_sum(x)
        x3 = tf_.elements_sum(to_tensorflow(x))
        x4 = torch_.elements_sum(to_torch(x))
        x5 = jnp_.elements_sum(to_jax(x))
        self.check_numbers_equal('elements_sum', [x1, x2, x3, x4, x5])

    def test_column_repeat(self):
        # (9) column_repeat(x: Matrix, n: int) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x = xc
        x1 = sympy_.column_repeat(to_sympy(x), n)
        x2 = np_.column_repeat(x, n)
        x3 = tf_.column_repeat(to_tensorflow(x), n)
        x4 = torch_.column_repeat(to_torch(x), n)
        x5 = jnp_.column_repeat(to_jax(x), n)
        self.check_arrays_equal('column_repeat', [x1, x2, x3, x4, x5])

    def test_row_repeat(self):
        # (10) row_repeat(xr: Matrix, m: int) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x = xr
        x1 = sympy_.row_repeat(to_sympy(x), n)
        x2 = np_.row_repeat(x, n)
        x3 = tf_.row_repeat(to_tensorflow(x), n)
        x4 = torch_.row_repeat(to_torch(x), n)
        x5 = jnp_.row_repeat(to_jax(x), n)
        self.check_arrays_equal('row_repeat', [x1, x2, x3, x4, x5])

    def test_columns_sum(self):
        # (11) columns_sum(X: Matrix) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x = X
        x1 = sympy_.columns_sum(to_sympy(x))
        x2 = np_.columns_sum(x)
        x3 = tf_.columns_sum(to_tensorflow(x))
        x4 = torch_.columns_sum(to_torch(x))
        x5 = jnp_.columns_sum(to_jax(x))
        self.check_arrays_equal('columns_sum', [x1, x2, x3, x4, x5])

    def test_rows_sum(self):
        # (12) rows_sum(X: Matrix) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x = X
        x1 = sympy_.rows_sum(to_sympy(x))
        x2 = np_.rows_sum(x)
        x3 = tf_.rows_sum(to_tensorflow(x))
        x4 = torch_.rows_sum(to_torch(x))
        x5 = jnp_.rows_sum(to_jax(x))
        self.check_arrays_equal('rows_sum', [x1, x2, x3, x4, x5])

    def test_columns_max(self):
        # (13) columns_max(X: Matrix) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x = X
        x1 = sympy_.columns_max(to_sympy(x))
        x2 = np_.columns_max(x)
        x3 = tf_.columns_max(to_tensorflow(x))
        x4 = torch_.columns_max(to_torch(x))
        x5 = jnp_.columns_max(to_jax(x))
        self.check_arrays_equal('columns_max', [x1, x2, x3, x4, x5])

    def test_rows_max(self):
        # (14) rows_max(X: Matrix) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x = X
        x1 = sympy_.rows_max(to_sympy(x))
        x2 = np_.rows_max(x)
        x3 = tf_.rows_max(to_tensorflow(x))
        x4 = torch_.rows_max(to_torch(x))
        x5 = jnp_.rows_max(to_jax(x))
        self.check_arrays_equal('rows_max', [x1, x2, x3, x4, x5])

    def test_columns_mean(self):
        # (15) columns_mean(X: Matrix) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x = X
        x1 = sympy_.columns_mean(to_sympy(x))
        x2 = np_.columns_mean(x)
        x3 = tf_.columns_mean(to_tensorflow(x))
        x4 = torch_.columns_mean(to_torch(x))
        x5 = jnp_.columns_mean(to_jax(x))
        self.check_arrays_equal('columns_mean', [x1, x2, x3, x4, x5])

    def test_rows_mean(self):
        # (16) rows_mean(X: Matrix) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x = X
        x1 = sympy_.rows_mean(to_sympy(x))
        x2 = np_.rows_mean(x)
        x3 = tf_.rows_mean(to_tensorflow(x))
        x4 = torch_.rows_mean(to_torch(x))
        x5 = jnp_.rows_mean(to_jax(x))
        self.check_arrays_equal('rows_mean', [x1, x2, x3, x4, x5])

    def test_apply(self):
        # (17) apply(f, X: Matrix) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x = X
        x1 = sympy_.apply(f, to_sympy(x))
        x2 = np_.apply(f, x)
        x3 = tf_.apply(f, to_tensorflow(x))
        x4 = torch_.apply(f, to_torch(x))
        x5 = jnp_.apply(f, to_jax(x))
        self.check_arrays_equal('apply', [x1, x2, x3, x4, x5])

    def test_exp(self):
        # (18) exp(X: Matrix) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x = X
        x1 = sympy_.exp(to_sympy(x))
        x2 = np_.exp(x)
        x3 = tf_.exp(to_tensorflow(x))
        x4 = torch_.exp(to_torch(x))
        x5 = jnp_.exp(to_jax(x))
        self.check_arrays_equal('exp', [x1, x2, x3, x4, x5])

    def test_log(self):
        # (19) log(X: Matrix) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x = X
        x1 = sympy_.log(to_sympy(x))
        x2 = np_.log(x)
        x3 = tf_.log(to_tensorflow(x))
        x4 = torch_.log(to_torch(x))
        x5 = jnp_.log(to_jax(x))
        self.check_arrays_equal('log', [x1, x2, x3, x4, x5])

    def test_inverse(self):
        # (20) inverse(X: Matrix) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x = X
        x1 = sympy_.inverse(to_sympy(x))
        x2 = np_.inverse(x)
        x3 = tf_.inverse(to_tensorflow(x))
        x4 = torch_.inverse(to_torch(x))
        x5 = jnp_.inverse(to_jax(x))
        self.check_arrays_equal('inverse', [x1, x2, x3, x4, x5])

    def test_square(self):
        # (21) square(X: Matrix) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x = X
        x1 = sympy_.square(to_sympy(x))
        x2 = np_.square(x)
        x3 = tf_.square(to_tensorflow(x))
        x4 = torch_.square(to_torch(x))
        x5 = jnp_.square(to_jax(x))
        self.check_arrays_equal('square', [x1, x2, x3, x4, x5])

    def test_sqrt(self):
        # (22) sqrt(X: Matrix) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x = X
        x1 = sympy_.sqrt(to_sympy(x))
        x2 = np_.sqrt(x)
        x3 = tf_.sqrt(to_tensorflow(x))
        x4 = torch_.sqrt(to_torch(x))
        x5 = jnp_.sqrt(to_jax(x))
        self.check_arrays_equal('sqrt', [x1, x2, x3, x4, x5])

    def test_power_minus_half(self):
        # (23) power_minus_half(X: Matrix) -> Matrix:
        m, n, f, X, Y, S, xc, xr = self.make_variables()

        x = X
        x1 = sympy_.power_minus_half(to_sympy(x))
        x2 = np_.power_minus_half(x)
        x3 = tf_.power_minus_half(to_tensorflow(x))
        x4 = torch_.power_minus_half(to_torch(x))
        x5 = jnp_.power_minus_half(to_jax(x))
        self.check_arrays_equal('power_minus_half', [x1, x2, x3, x4, x5])


class TestConversions(TestCase):

    def test_to_row_col(self):
        xc = np.array([
            [9],
            [3],
            [12],
        ], dtype=float)

        xr = np.array([
            [11, 2, 3]
        ], dtype=float)

        x = np.array([11, 2, 3], dtype=float)

        self.assertEqual(np_.to_row(x).shape, (1, 3))
        self.assertEqual(np_.to_row(xr).shape, (1, 3))

        self.assertEqual(np_.to_col(x).shape, (3, 1))
        self.assertEqual(np_.to_col(xc).shape, (3, 1))


if __name__ == '__main__':
    import unittest
    unittest.main()
