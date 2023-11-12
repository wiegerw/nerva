#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase
import numpy as np
from mlps.nerva_sympy.matrix_operations import *
from mlps.nerva_sympy.softmax_functions import *
import mlps.nerva_numpy.softmax_functions as np_
import mlps.nerva_tensorflow.softmax_functions as tf_
import mlps.nerva_torch.softmax_functions as torch_
import mlps.nerva_sympy.softmax_functions as sympy_
import mlps.nerva_jax.softmax_functions as jnp_
import nervalibcolwise as eigen_
from mlps.tests.test_utilities import to_numpy, to_sympy, to_tensorflow, to_torch, to_jax, to_eigen, check_arrays_equal, \
    check_numbers_equal

Matrix = sp.Matrix

#-------------------------------------#
# alternative implementations of softmax functions
#-------------------------------------#

def softmax_colwise1(X: Matrix) -> Matrix:
    D, N = X.shape

    def softmax(x):
        e = exp(x)
        return e / sum(e)

    return Matrix([softmax(X.col(j)).T for j in range(N)]).T


def softmax_colwise_jacobian1(x: Matrix) -> Matrix:
    return jacobian(softmax_colwise1(x), x)


def log_softmax_colwise1(X: Matrix) -> Matrix:
    return log(softmax_colwise(X))


def log_softmax_colwise_jacobian1(x: Matrix) -> Matrix:
    return jacobian(log_softmax_colwise(x), x)


def softmax_rowwise1(X: Matrix) -> Matrix:
    N, D = X.shape

    def softmax(x):
        e = exp(x)
        return e / sum(e)

    return join_rows([softmax(X.row(i)) for i in range(N)])


def softmax_rowwise2(X: Matrix) -> Matrix:
    return softmax_colwise(X.T).T


def softmax_rowwise_jacobian1(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    return jacobian(softmax_rowwise(x), x)


def softmax_rowwise_jacobian2(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    return softmax_colwise_jacobian(x.T).T


def log_softmax_rowwise1(X: Matrix) -> Matrix:
    return log(softmax_rowwise(X))


def log_softmax_rowwise2(X: Matrix) -> Matrix:
    return log_softmax_colwise(X.T).T


def log_softmax_rowwise_jacobian1(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    return jacobian(log_softmax_rowwise(x), x)


def log_softmax_rowwise_jacobian2(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    return log_softmax_colwise_jacobian(x.T)


class TestSoftmaxColwise(TestCase):
    def test_softmax_colwise(self):
        D = 3
        N = 2
        X = Matrix(sp.symarray('x', (D, N), real=True))

        y1 = softmax_colwise(X)
        y2 = softmax_colwise1(X)
        y3 = stable_softmax_colwise(X)
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(D, N))
        self.assertEqual(sp.simplify(y1 - y3), sp.zeros(D, N))

        y1 = log_softmax_colwise(X)
        y2 = log_softmax_colwise1(X)
        y3 = stable_log_softmax_colwise(X)
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(D, N))
        self.assertEqual(sp.simplify(y1 - y3), sp.zeros(D, N))

    def test_softmax_colwise_jacobian(self):
        x = Matrix(sp.symbols('x y z'), real=True)
        D, N = x.shape

        y1 = sp.simplify(softmax_colwise_jacobian(x))
        y2 = sp.simplify(softmax_colwise_jacobian1(x))
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(D, D))

    def test_log_softmax_colwise_jacobian(self):
        x = Matrix(sp.symbols('x y z'), real=True)
        D, N = x.shape

        y1 = sp.simplify(log_softmax_colwise_jacobian(x))
        y2 = sp.simplify(log_softmax_colwise_jacobian1(x))
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(D, D))


class TestSoftmaxRowwise(TestCase):
    def test_softmax_rowwise(self):
        D = 3
        N = 2
        X = Matrix(sp.symarray('x', (N, D), real=True))

        y1 = softmax_rowwise(X)
        y2 = softmax_rowwise1(X)
        y3 = softmax_rowwise2(X)
        y4 = stable_softmax_rowwise(X)
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(N, D))
        self.assertEqual(sp.simplify(y1 - y3), sp.zeros(N, D))
        self.assertEqual(sp.simplify(y1 - y4), sp.zeros(N, D))

        y1 = log_softmax_rowwise(X)
        y2 = log_softmax_rowwise1(X)
        y3 = log_softmax_rowwise2(X)
        y4 = stable_log_softmax_rowwise(X)
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(N, D))
        self.assertEqual(sp.simplify(y1 - y3), sp.zeros(N, D))
        self.assertEqual(sp.simplify(y1 - y4), sp.zeros(N, D))

    def test_softmax_rowwise_jacobian(self):
        x = Matrix(sp.symbols('x y z'), real=True).T
        N, D = x.shape

        y1 = sp.simplify(softmax_rowwise_jacobian(x))
        y2 = sp.simplify(softmax_rowwise_jacobian1(x))
        y3 = sp.simplify(softmax_rowwise_jacobian2(x))
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(D, D))
        self.assertEqual(sp.simplify(y1 - y3), sp.zeros(D, D))

    def test_log_softmax_rowwise_jacobian(self):
        x = Matrix(sp.symbols('x y z'), real=True).T
        N, D = x.shape

        y1 = sp.simplify(log_softmax_rowwise_jacobian(x))
        y2 = sp.simplify(log_softmax_rowwise_jacobian1(x))
        y3 = sp.simplify(log_softmax_rowwise_jacobian2(x))
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(D, D))
        self.assertEqual(sp.simplify(y1 - y3), sp.zeros(D, D))


class TestSoftmaxColwiseValues(TestCase):
    def make_variables(self):
        X = np.array([
            [1, 2, 3],
            [7, 3, 4]
        ], dtype=np.float32)

        x = np.array([
            [9],
            [3],
            [12],
        ], dtype=np.float32)

        return X, x

    def _test_softmax(self, function_name, x):
        f_sympy = getattr(sympy_, function_name)
        f_numpy = getattr(np_, function_name)
        f_tensorflow = getattr(tf_, function_name)
        f_torch = getattr(torch_, function_name)
        f_jax = getattr(jnp_, function_name)
        f_eigen = getattr(eigen_, function_name)

        x1 = f_sympy(to_sympy(x))
        x2 = f_numpy(to_numpy(x))
        x3 = f_tensorflow(to_tensorflow(x))
        x4 = f_torch(to_torch(x))
        x5 = f_jax(to_jax(x))
        x6 = f_eigen(to_eigen(x))

        if isinstance(x1, sp.Matrix):
            check_arrays_equal(self, function_name, [x1, x2, x3, x4, x5, x6])
        else:
            check_numbers_equal(self, function_name, [x1, x2, x3, x4, x5, x6])

    def test_all(self):
        X, x = self.make_variables()
        self._test_softmax('softmax_colwise', X)
        self._test_softmax('softmax_colwise_jacobian', x)
        self._test_softmax('stable_softmax_colwise', X)
        self._test_softmax('log_softmax_colwise', X)
        self._test_softmax('stable_log_softmax_colwise', X)
        self._test_softmax('log_softmax_colwise_jacobian', x)


class TestSoftmaxRowwiseValues(TestCase):
    def make_variables(self):
        X = np.array([
            [1, 2, 3],
            [7, 3, 4]
        ], dtype=np.float32)

        x = np.array([
            [11, 2, 3]
        ], dtype=np.float32)

        return X, x

    def _test_softmax(self, function_name, x):
        f_sympy = getattr(sympy_, function_name)
        f_numpy = getattr(np_, function_name)
        f_tensorflow = getattr(tf_, function_name)
        f_torch = getattr(torch_, function_name)
        f_jax = getattr(jnp_, function_name)
        f_eigen = getattr(eigen_, function_name)

        x1 = f_sympy(to_sympy(x))
        x2 = f_numpy(to_numpy(x))
        x3 = f_tensorflow(to_tensorflow(x))
        x4 = f_torch(to_torch(x))
        x5 = f_jax(to_jax(x))
        x6 = f_eigen(to_eigen(x))

        if isinstance(x1, sp.Matrix):
            check_arrays_equal(self, function_name, [x1, x2, x3, x4, x5, x6])
        else:
            check_numbers_equal(self, function_name, [x1, x2, x3, x4, x5, x6])

    def test_all(self):
        X, x = self.make_variables()
        self._test_softmax('softmax_rowwise', X)
        self._test_softmax('softmax_rowwise_jacobian', x)
        self._test_softmax('stable_softmax_rowwise', X)
        self._test_softmax('log_softmax_rowwise', X)
        self._test_softmax('log_softmax_rowwise_jacobian', x)
        self._test_softmax('stable_log_softmax_rowwise', X)


if __name__ == '__main__':
    import unittest
    unittest.main()
