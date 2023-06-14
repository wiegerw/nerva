#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase

import numpy as np

from symbolic.matrix_operations_sympy import substitute
from symbolic.utilities import to_numpy, to_sympy, to_torch, to_tensorflow, matrix, equal_matrices, \
    instantiate_one_hot_colwise, instantiate_one_hot_rowwise
import symbolic.loss_functions_numpy as np_
import symbolic.loss_functions_tensorflow as tf_
import symbolic.loss_functions_torch as torch_
import symbolic.loss_functions_sympy as sympy_


class TestCaseLossFunction(TestCase):
    def make_variables(self):
        raise NotImplementedError

    def _test_loss_function(self, function_name: str, one_hot=False):
        K, y, t, Y, T = self.make_variables()

        # retrieve functions by name
        loss_value = getattr(sympy_, function_name)
        loss_gradient = getattr(sympy_, f'{function_name}_gradient')
        Loss_value = getattr(sympy_, function_name.capitalize())
        Loss_gradient = getattr(sympy_, f'{function_name.capitalize()}_gradient')

        loss = loss_value(y, t)
        Dy1 = loss_gradient(y, t)
        Dy2 = sympy_.diff(loss, y)
        self.assertTrue(equal_matrices(Dy1, Dy2))

        loss = Loss_value(Y, T)
        DY1 = Loss_gradient(Y, T)
        DY2 = sympy_.diff(loss, Y)
        self.assertTrue(equal_matrices(DY1, DY2))

        if one_hot:
            loss_gradient_one_hot = getattr(sympy_, f'{function_name}_gradient_one_hot')
            Loss_gradient_one_hot = getattr(sympy_, f'{function_name.capitalize()}_gradient_one_hot')

            # test with a one-hot encoded vector t0
            t0 = instantiate_one_hot_colwise(t) if 'colwise' in function_name else instantiate_one_hot_rowwise(t)
            Dy1_a = substitute(loss_gradient_one_hot(y, t), (t, t0))
            Dy2_a = substitute(Dy2, (t, t0))
            self.assertTrue(equal_matrices(Dy1_a, Dy2_a))

            # test with a one-hot encoded matrix T0
            T0 = instantiate_one_hot_colwise(T) if 'colwise' in function_name else instantiate_one_hot_rowwise(T)
            DY1_a = substitute(Loss_gradient_one_hot(Y, T), (T, T0))
            DY2_a = substitute(DY2, (T, T0))
            self.assertTrue(equal_matrices(DY1_a, DY2_a))


class TestColwiseLossFunctionGradients(TestCaseLossFunction):
    def make_variables(self):
        K = 3
        N = 2
        y = matrix('y', K, 1)
        t = matrix('t', K, 1)
        Y = matrix('Y', K, N)
        T = matrix('T', K, N)
        return K, y, t, Y, T

    def test_squared_error_loss_colwise(self):
        self._test_loss_function('squared_error_loss_colwise')

    def test_mean_squared_error_loss_colwise(self):
        self._test_loss_function('mean_squared_error_loss_colwise')

    def test_cross_entropy_loss_colwise(self):
        self._test_loss_function('cross_entropy_loss_colwise')

    def test_softmax_cross_entropy_loss_colwise(self):
        self._test_loss_function('softmax_cross_entropy_loss_colwise', one_hot=True)

    def test_stable_softmax_cross_entropy_loss_colwise(self):
        self._test_loss_function('stable_softmax_cross_entropy_loss_colwise', one_hot=True)

    def test_logistic_cross_entropy_loss_colwise(self):
        self._test_loss_function('logistic_cross_entropy_loss_colwise')

    def test_negative_log_likelihood_loss_colwise(self):
        self._test_loss_function('negative_log_likelihood_loss_colwise')


class TestRowwiseLossFunctionGradients(TestCaseLossFunction):
    def make_variables(self):
        K = 3
        N = 2
        y = matrix('y', 1, K)
        t = matrix('t', 1, K)
        Y = matrix('Y', N, K)
        T = matrix('T', N, K)
        return K, y, t, Y, T

    def test_squared_error_loss_rowwise(self):
        self._test_loss_function('squared_error_loss_rowwise')

    def test_mean_squared_error_loss_rowwise(self):
        self._test_loss_function('mean_squared_error_loss_rowwise')

    def test_cross_entropy_loss_rowwise(self):
        self._test_loss_function('cross_entropy_loss_rowwise')

    def test_softmax_cross_entropy_loss_rowwise(self):
        self._test_loss_function('softmax_cross_entropy_loss_rowwise', one_hot=True)

    def test_stable_softmax_cross_entropy_loss_rowwise(self):
        self._test_loss_function('stable_softmax_cross_entropy_loss_rowwise', one_hot=True)

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
        yc = np.array([
            [9],
            [3],
            [12],
        ], dtype=float)

        tc = np.array([
            [0],
            [0],
            [1],
        ], dtype=float)

        yr = np.array([
            [11, 2, 3]
        ], dtype=float)

        tr = np.array([
            [0, 1, 0]
        ], dtype=float)

        Y = np.array([
            [1, 2, 3],
            [7, 3, 4]
        ], dtype=float)

        T = np.array([
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=float)

        return yc, tc, yr, tr, Y, T

    def _test_loss_function(self, function_name: str):
        yc, tc, yr, tr, Y, T = self.make_variables()

        # test loss on vectors
        name = function_name
        f_sympy = getattr(sympy_, name)
        f_numpy = getattr(np_, name)
        f_tensorflow = getattr(tf_, name)
        f_torch = getattr(torch_, name)
        y, t = (yc, tc) if 'colwise' in name else (yr, tr)
        x1 = f_sympy(to_sympy(y), to_sympy(t))
        x2 = f_numpy(to_numpy(y), to_numpy(t))
        x3 = f_tensorflow(to_tensorflow(y), to_tensorflow(t))
        x4 = f_torch(to_torch(y), to_torch(t))
        self.check_numbers_equal(function_name, [x1, x2, x3, x4])

        # test loss gradient on vectors
        name = f'{function_name}_gradient'
        f_sympy = getattr(sympy_, name)
        f_numpy = getattr(np_, name)
        f_tensorflow = getattr(tf_, name)
        f_torch = getattr(torch_, name)
        y, t = (yc, tc) if 'colwise' in name else (yr, tr)
        x1 = f_sympy(to_sympy(y), to_sympy(t))
        x2 = f_numpy(to_numpy(y), to_numpy(t))
        x3 = f_tensorflow(to_tensorflow(y), to_tensorflow(t))
        x4 = f_torch(to_torch(y), to_torch(t))
        self.check_arrays_equal(function_name, [x1, x2, x3, x4])

        # test loss on matrices
        name = function_name.capitalize()
        f_sympy = getattr(sympy_, name)
        f_numpy = getattr(np_, name)
        f_tensorflow = getattr(tf_, name)
        f_torch = getattr(torch_, name)
        y, t = (Y.T, T.T) if 'colwise' in name else (Y, T)
        x1 = f_sympy(to_sympy(y), to_sympy(t))
        x2 = f_numpy(to_numpy(y), to_numpy(t))
        x3 = f_tensorflow(to_tensorflow(y), to_tensorflow(t))
        x4 = f_torch(to_torch(y), to_torch(t))
        self.check_numbers_equal(function_name, [x1, x2, x3, x4])

        # test loss gradient on matrices
        name = f'{function_name.capitalize()}_gradient'
        f_sympy = getattr(sympy_, name)
        f_numpy = getattr(np_, name)
        f_tensorflow = getattr(tf_, name)
        f_torch = getattr(torch_, name)
        y, t = (Y.T, T.T) if 'colwise' in name else (Y, T)
        x1 = f_sympy(to_sympy(y), to_sympy(t))
        x2 = f_numpy(to_numpy(y), to_numpy(t))
        x3 = f_tensorflow(to_tensorflow(y), to_tensorflow(t))
        x4 = f_torch(to_torch(y), to_torch(t))
        self.check_arrays_equal(function_name, [x1, x2, x3, x4])

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
