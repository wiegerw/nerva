#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import random
from unittest import TestCase
import numpy as np
import sympy as sp
from mlps.nerva_sympy.matrix_operations import substitute, diff
import mlps.nerva_numpy.loss_functions as np_
import mlps.nerva_tensorflow.loss_functions as tf_
import mlps.nerva_torch.loss_functions as torch_
import mlps.nerva_sympy.loss_functions as sympy_
import mlps.nerva_jax.loss_functions as jnp_
import nervalibcolwise as eigen_
from mlps.tests.test_utilities import to_numpy, to_sympy, to_torch, to_tensorflow, to_jax, to_eigen
from mlps.tests.sympy_utilities import matrix, equal_matrices


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


class TestColwiseLossFunctionGradients(TestCase):
    def make_variables(self):
        K = 3
        N = 2
        y = matrix('y', K, 1)
        t = matrix('t', K, 1)
        Y = matrix('Y', K, N)
        T = matrix('T', K, N)
        return K, y, t, Y, T

    def _test_loss_function(self, function_name: str):
        K, y, t, Y, T = self.make_variables()

        # retrieve functions by name
        loss_value = getattr(sympy_, function_name)
        loss_gradient = getattr(sympy_, f'{function_name}_gradient')
        Loss_value = getattr(sympy_, function_name.capitalize())
        Loss_gradient = getattr(sympy_, f'{function_name.capitalize()}_gradient')

        loss = loss_value(y, t)
        Dy1 = loss_gradient(y, t)
        Dy2 = diff(loss, y)
        self.assertTrue(equal_matrices(Dy1, Dy2))

        loss = Loss_value(Y, T)
        DY1 = Loss_gradient(Y, T)
        DY2 = diff(loss, Y)
        self.assertTrue(equal_matrices(DY1, DY2))

    def _test_loss_function_one_hot(self, function_name: str):
        K, y, t, Y, T = self.make_variables()

        # retrieve functions by name
        loss_gradient = getattr(sympy_, f'{function_name}_gradient')
        Loss_gradient = getattr(sympy_, f'{function_name.capitalize()}_gradient')
        loss_gradient_one_hot = getattr(sympy_, f'{function_name}_gradient_one_hot')
        Loss_gradient_one_hot = getattr(sympy_, f'{function_name.capitalize()}_gradient_one_hot')

        Dy = loss_gradient(y, t)
        DY = Loss_gradient(Y, T)

        # test with a one-hot encoded vector t0
        t_0 = instantiate_one_hot_colwise(t)
        Dy1 = substitute(loss_gradient_one_hot(y, t), (t, t_0))
        Dy2 = substitute(Dy, (t, t_0))
        self.assertTrue(equal_matrices(Dy1, Dy2))

        # test with a one-hot encoded matrix T0
        T0 = instantiate_one_hot_colwise(T)
        DY1 = substitute(Loss_gradient_one_hot(Y, T), (T, T0))
        DY2 = substitute(DY, (T, T0))
        self.assertTrue(equal_matrices(DY1, DY2))

    def test_squared_error_loss_colwise(self):
        self._test_loss_function('squared_error_loss_colwise')

    def test_mean_squared_error_loss_colwise(self):
        self._test_loss_function('mean_squared_error_loss_colwise')

    def test_cross_entropy_loss_colwise(self):
        self._test_loss_function('cross_entropy_loss_colwise')

    def test_softmax_cross_entropy_loss_colwise(self):
        self._test_loss_function('softmax_cross_entropy_loss_colwise')
        self._test_loss_function_one_hot('softmax_cross_entropy_loss_colwise')

    def test_stable_softmax_cross_entropy_loss_colwise(self):
        self._test_loss_function('stable_softmax_cross_entropy_loss_colwise')
        self._test_loss_function_one_hot('stable_softmax_cross_entropy_loss_colwise')

    def test_logistic_cross_entropy_loss_colwise(self):
        self._test_loss_function('logistic_cross_entropy_loss_colwise')

    def test_negative_log_likelihood_loss_colwise(self):
        self._test_loss_function('negative_log_likelihood_loss_colwise')


class TestRowwiseLossFunctionGradients(TestCase):
    def make_variables(self):
        K = 3
        N = 2
        y = matrix('y', 1, K)
        t = matrix('t', 1, K)
        Y = matrix('Y', N, K)
        T = matrix('T', N, K)
        return K, y, t, Y, T

    def _test_loss_function(self, function_name: str):
        K, y, t, Y, T = self.make_variables()

        # retrieve functions by name
        loss_value = getattr(sympy_, function_name)
        loss_gradient = getattr(sympy_, f'{function_name}_gradient')
        Loss_value = getattr(sympy_, function_name.capitalize())
        Loss_gradient = getattr(sympy_, f'{function_name.capitalize()}_gradient')

        loss = loss_value(y, t)
        Dy1 = loss_gradient(y, t)
        Dy2 = diff(loss, y)
        self.assertTrue(equal_matrices(Dy1, Dy2))

        loss = Loss_value(Y, T)
        DY1 = Loss_gradient(Y, T)
        DY2 = diff(loss, Y)
        self.assertTrue(equal_matrices(DY1, DY2))

    def _test_loss_function_one_hot(self, function_name: str):
        K, y, t, Y, T = self.make_variables()

        # retrieve functions by name
        loss_gradient = getattr(sympy_, f'{function_name}_gradient')
        Loss_gradient = getattr(sympy_, f'{function_name.capitalize()}_gradient')
        loss_gradient_one_hot = getattr(sympy_, f'{function_name}_gradient_one_hot')
        Loss_gradient_one_hot = getattr(sympy_, f'{function_name.capitalize()}_gradient_one_hot')

        Dy = loss_gradient(y, t)
        DY = Loss_gradient(Y, T)

        # test with a one-hot encoded vector t0
        t_0 = instantiate_one_hot_rowwise(t)
        Dy1 = substitute(loss_gradient_one_hot(y, t), (t, t_0))
        Dy2 = substitute(Dy, (t, t_0))
        self.assertTrue(equal_matrices(Dy1, Dy2))

        # test with a one-hot encoded matrix T0
        T0 = instantiate_one_hot_rowwise(T)
        DY1 = substitute(Loss_gradient_one_hot(Y, T), (T, T0))
        DY2 = substitute(DY, (T, T0))
        self.assertTrue(equal_matrices(DY1, DY2))

    def test_squared_error_loss_rowwise(self):
        self._test_loss_function('squared_error_loss_rowwise')

    def test_mean_squared_error_loss_rowwise(self):
        self._test_loss_function('mean_squared_error_loss_rowwise')

    def test_cross_entropy_loss_rowwise(self):
        self._test_loss_function('cross_entropy_loss_rowwise')

    def test_softmax_cross_entropy_loss_rowwise(self):
        self._test_loss_function('softmax_cross_entropy_loss_rowwise')
        self._test_loss_function_one_hot('softmax_cross_entropy_loss_rowwise')

    def test_stable_softmax_cross_entropy_loss_rowwise(self):
        self._test_loss_function('stable_softmax_cross_entropy_loss_rowwise')
        self._test_loss_function_one_hot('stable_softmax_cross_entropy_loss_rowwise')

    def test_logistic_cross_entropy_loss_rowwise(self):
        self._test_loss_function('logistic_cross_entropy_loss_rowwise')

    def test_negative_log_likelihood_loss_rowwise(self):
        self._test_loss_function('negative_log_likelihood_loss_rowwise')


class TestColwiseLossFunctionValues(TestCase):
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
        y = np.array([
            [9],
            [3],
            [12],
        ], dtype=np.float32)

        t = np.array([
            [0],
            [0],
            [1],
        ], dtype=np.float32)

        Y = np.array([
            [1, 7],
            [2, 3],
            [3, 4],
        ], dtype=np.float32)

        T = np.array([
            [1, 0],
            [0, 1],
            [0, 0]
        ], dtype=np.float32)

        return y, t, Y, T

    def _test_loss_function(self, function_name: str):
        y, t, Y, T = self.make_variables()

        print('=== test loss on vectors ===')
        name = function_name
        f_sympy = getattr(sympy_, name)
        f_numpy = getattr(np_, name)
        f_tensorflow = getattr(tf_, name)
        f_torch = getattr(torch_, name)
        f_jax = getattr(jnp_, name)
        f_eigen = getattr(eigen_, name)
        x1 = f_sympy(to_sympy(y), to_sympy(t))
        x2 = f_numpy(to_numpy(y), to_numpy(t))
        x3 = f_tensorflow(to_tensorflow(y), to_tensorflow(t))
        x4 = f_torch(to_torch(y), to_torch(t))
        x5 = f_jax(to_jax(y), to_jax(t))
        x6 = f_eigen(to_eigen(y), to_eigen(t))
        self.check_numbers_equal(function_name, [x1, x2, x3, x4, x5, x6])

        print('=== test loss gradient on vectors ===')
        name = f'{function_name}_gradient'
        f_sympy = getattr(sympy_, name)
        f_numpy = getattr(np_, name)
        f_tensorflow = getattr(tf_, name)
        f_torch = getattr(torch_, name)
        f_jax = getattr(jnp_, name)
        f_eigen = getattr(eigen_, name)
        x1 = f_sympy(to_sympy(y), to_sympy(t))
        x2 = f_numpy(to_numpy(y), to_numpy(t))
        x3 = f_tensorflow(to_tensorflow(y), to_tensorflow(t))
        x4 = f_torch(to_torch(y), to_torch(t))
        x5 = f_jax(to_jax(y), to_jax(t))
        x6 = f_eigen(to_eigen(y), to_eigen(t))
        self.check_arrays_equal(function_name, [x1, x2, x3, x4, x5, x6])

        print('=== test loss on matrices ===')
        name = function_name.capitalize()
        f_sympy = getattr(sympy_, name)
        f_numpy = getattr(np_, name)
        f_tensorflow = getattr(tf_, name)
        f_torch = getattr(torch_, name)
        f_jax = getattr(jnp_, name)
        f_eigen = getattr(eigen_, name)
        x1 = f_sympy(to_sympy(Y), to_sympy(T))
        x2 = f_numpy(to_numpy(Y), to_numpy(T))
        x3 = f_tensorflow(to_tensorflow(Y), to_tensorflow(T))
        x4 = f_torch(to_torch(Y), to_torch(T))
        x5 = f_jax(to_jax(Y), to_jax(T))
        x6 = f_eigen(to_eigen(Y), to_eigen(T))
        self.check_numbers_equal(function_name, [x1, x2, x3, x4, x5, x6])

        print('=== test loss gradient on matrices ===')
        name = f'{function_name.capitalize()}_gradient'
        f_sympy = getattr(sympy_, name)
        f_numpy = getattr(np_, name)
        f_tensorflow = getattr(tf_, name)
        f_torch = getattr(torch_, name)
        f_jax = getattr(jnp_, name)
        f_eigen = getattr(eigen_, name)
        x1 = f_sympy(to_sympy(Y), to_sympy(T))
        x2 = f_numpy(to_numpy(Y), to_numpy(T))
        x3 = f_tensorflow(to_tensorflow(Y), to_tensorflow(T))
        x4 = f_torch(to_torch(Y), to_torch(T))
        x5 = f_jax(to_jax(Y), to_jax(T))
        x6 = f_eigen(to_eigen(Y), to_eigen(T))
        self.check_arrays_equal(function_name, [x1, x2, x3, x4, x5, x6])

    def test_squared_error_loss_colwise(self):
        self._test_loss_function('squared_error_loss_colwise')

    def test_mean_squared_error_loss_colwise(self):
        self._test_loss_function('mean_squared_error_loss_colwise')

    def test_cross_entropy_loss_colwise(self):
        self._test_loss_function('cross_entropy_loss_colwise')

    def test_softmax_cross_entropy_loss_colwise(self):
        self._test_loss_function('softmax_cross_entropy_loss_colwise')

    def test_stable_softmax_cross_entropy_loss_colwise(self):
        self._test_loss_function('stable_softmax_cross_entropy_loss_colwise')

    def test_logistic_cross_entropy_loss_colwise(self):
        self._test_loss_function('logistic_cross_entropy_loss_colwise')

    def test_negative_log_likelihood_loss_colwise(self):
        self._test_loss_function('negative_log_likelihood_loss_colwise')


class TestRowwiseLossFunctionValues(TestCase):
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
        y = np.array([
            [11, 2, 3]
        ], dtype=np.float32)

        t = np.array([
            [0, 1, 0]
        ], dtype=np.float32)

        Y = np.array([
            [1, 2, 3],
            [7, 3, 4]
        ], dtype=np.float32)

        T = np.array([
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=np.float32)

        return y, t, Y, T

    def _test_loss_function(self, function_name: str):
        y, t, Y, T = self.make_variables()

        print('=== test loss on vectors ===')
        name = function_name
        f_sympy = getattr(sympy_, name)
        f_numpy = getattr(np_, name)
        f_tensorflow = getattr(tf_, name)
        f_torch = getattr(torch_, name)
        f_jax = getattr(jnp_, name)
        f_eigen = getattr(eigen_, name)
        x1 = f_sympy(to_sympy(y), to_sympy(t))
        x2 = f_numpy(to_numpy(y), to_numpy(t))
        x3 = f_tensorflow(to_tensorflow(y), to_tensorflow(t))
        x4 = f_torch(to_torch(y), to_torch(t))
        x5 = f_jax(to_jax(y), to_jax(t))
        x6 = f_eigen(to_eigen(y), to_eigen(t))
        self.check_numbers_equal(function_name, [x1, x2, x3, x4, x5, x6])

        print('=== test loss gradient on vectors ===')
        name = f'{function_name}_gradient'
        f_sympy = getattr(sympy_, name)
        f_numpy = getattr(np_, name)
        f_tensorflow = getattr(tf_, name)
        f_torch = getattr(torch_, name)
        f_jax = getattr(jnp_, name)
        f_eigen = getattr(eigen_, name)
        x1 = f_sympy(to_sympy(y), to_sympy(t))
        x2 = f_numpy(to_numpy(y), to_numpy(t))
        x3 = f_tensorflow(to_tensorflow(y), to_tensorflow(t))
        x4 = f_torch(to_torch(y), to_torch(t))
        x5 = f_jax(to_jax(y), to_jax(t))
        x6 = f_eigen(to_eigen(y), to_eigen(t))
        self.check_arrays_equal(function_name, [x1, x2, x3, x4, x5, x6])

        print('=== test loss on matrices ===')
        name = function_name.capitalize()
        f_sympy = getattr(sympy_, name)
        f_numpy = getattr(np_, name)
        f_tensorflow = getattr(tf_, name)
        f_torch = getattr(torch_, name)
        f_jax = getattr(jnp_, name)
        f_eigen = getattr(eigen_, name)
        x1 = f_sympy(to_sympy(Y), to_sympy(T))
        x2 = f_numpy(to_numpy(Y), to_numpy(T))
        x3 = f_tensorflow(to_tensorflow(Y), to_tensorflow(T))
        x4 = f_torch(to_torch(Y), to_torch(T))
        x5 = f_jax(to_jax(Y), to_jax(T))
        x6 = f_eigen(to_eigen(Y), to_eigen(T))
        self.check_numbers_equal(function_name, [x1, x2, x3, x4, x5, x6])

        print('=== test loss gradient on matrices ===')
        name = f'{function_name.capitalize()}_gradient'
        f_sympy = getattr(sympy_, name)
        f_numpy = getattr(np_, name)
        f_tensorflow = getattr(tf_, name)
        f_torch = getattr(torch_, name)
        f_jax = getattr(jnp_, name)
        f_eigen = getattr(eigen_, name)
        x1 = f_sympy(to_sympy(Y), to_sympy(T))
        x2 = f_numpy(to_numpy(Y), to_numpy(T))
        x3 = f_tensorflow(to_tensorflow(Y), to_tensorflow(T))
        x4 = f_torch(to_torch(Y), to_torch(T))
        x5 = f_jax(to_jax(Y), to_jax(T))
        x6 = f_eigen(to_eigen(Y), to_eigen(T))
        self.check_arrays_equal(function_name, [x1, x2, x3, x4, x5, x6])

    def test_squared_error_loss_rowwise(self):
        self._test_loss_function('squared_error_loss_rowwise')

    def test_mean_squared_error_loss_rowwise(self):
        self._test_loss_function('mean_squared_error_loss_rowwise')

    def test_cross_entropy_loss_rowwise(self):
        self._test_loss_function('cross_entropy_loss_rowwise')

    def test_softmax_cross_entropy_loss_rowwise(self):
        self._test_loss_function('softmax_cross_entropy_loss_rowwise')

    def test_stable_softmax_cross_entropy_loss_rowwise(self):
        self._test_loss_function('stable_softmax_cross_entropy_loss_rowwise')

    def test_logistic_cross_entropy_loss_rowwise(self):
        self._test_loss_function('logistic_cross_entropy_loss_rowwise')

    def test_negative_log_likelihood_loss_rowwise(self):
        self._test_loss_function('negative_log_likelihood_loss_rowwise')


if __name__ == '__main__':
    import unittest
    unittest.main()
