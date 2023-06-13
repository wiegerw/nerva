#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase

import numpy as np

from symbolic.matrix_operations_sympy import substitute, elements_sum
from symbolic.utilities import to_numpy, to_sympy, to_torch, to_tensorflow, matrix, equal_matrices, \
    instantiate_one_hot_colwise, instantiate_one_hot_rowwise
import symbolic.loss_functions_numpy as np_
import symbolic.loss_functions_tensorflow as tf_
import symbolic.loss_functions_torch as torch_
import symbolic.loss_functions_sympy as sympy_


class TestColwiseLossFunctionGradients(TestCase):
    def make_variables(self):
        K = 3
        N = 2
        y = matrix('y', K, 1)
        t = matrix('t', K, 1)
        Y = matrix('Y', K, N)
        T = matrix('T', K, N)
        return K, y, t, Y, T

    def _test_loss_function(self, f):
        K, y, t, Y, T = self.make_variables()
        loss = f.vector_value(y, t)
        Dy1 = f.vector_gradient(y, t)
        Dy2 = sympy_.diff(loss, y)
        self.assertTrue(equal_matrices(Dy1, Dy2))

        loss = f.value(Y, T)
        DY1 = f.gradient(Y, T)
        DY2 = sympy_.diff(loss, Y)
        self.assertTrue(equal_matrices(DY1, DY2))

    def _test_loss_function_one_hot(self, f, colwise=True):
        K, y, t, Y, T = self.make_variables()
        loss = f.vector_value(y, t)
        Dy1 = f.vector_gradient(y, t)
        Dy2 = sympy_.diff(loss, y)
        self.assertTrue(equal_matrices(Dy1, Dy2))

        loss = f.value(Y, T)
        DY1 = f.gradient(Y, T)
        DY2 = sympy_.diff(loss, Y)
        self.assertTrue(equal_matrices(DY1, DY2))

        # test with a one-hot encoded vector t0
        t0 = instantiate_one_hot_colwise(t) if colwise else instantiate_one_hot_rowwise(t)
        Dy1_a = substitute(f.vector_gradient_one_hot(y, t), (t, t0))
        Dy2_a = substitute(Dy2, (t, t0))
        self.assertTrue(equal_matrices(Dy1_a, Dy2_a))

        # test with a one-hot encoded matrix T0
        T0 = instantiate_one_hot_colwise(T) if colwise else instantiate_one_hot_rowwise(T)
        DY1_a = substitute(f.gradient_one_hot(Y, T), (T, T0))
        DY2_a = substitute(DY2, (T, T0))
        self.assertTrue(equal_matrices(DY1_a, DY2_a))


    def test_squared_error_loss(self):
        f = sympy_.SquaredErrorLossColwise()
        self._test_loss_function(f)

    def test_mean_squared_error_loss(self):
        f = sympy_.MeanSquaredErrorLossColwise()
        self._test_loss_function(f)

    def test_cross_entropy_loss_loss(self):
        f = sympy_.CrossEntropyLossColwise()
        self._test_loss_function(f)

    def test_softmax_cross_entropy_loss(self):
        f = sympy_.SoftmaxCrossEntropyLossColwise()
        self._test_loss_function_one_hot(f)

    def test_stable_softmax_cross_entropy_loss(self):
        f = sympy_.StableSoftmaxCrossEntropyLossColwise()
        self._test_loss_function_one_hot(f)

    def test_logistic_cross_entropy_loss(self):
        f = sympy_.LogisticCrossEntropyLossColwise()
        self._test_loss_function(f)

    def test_negative_log_likelihood_loss(self):
        f = sympy_.NegativeLogLikelihoodLossColwise()
        self._test_loss_function(f)


class TestRowwiseLossFunctionGradients(TestColwiseLossFunctionGradients):
    def make_variables(self):
        K = 3
        N = 2
        y = matrix('y', 1, K)
        t = matrix('t', 1, K)
        Y = matrix('Y', N, K)
        T = matrix('T', N, K)
        return K, y, t, Y, T

    def test_squared_error_loss(self):
        f = sympy_.SquaredErrorLossRowwise()
        self._test_loss_function(f)

    def test_mean_squared_error_loss(self):
        f = sympy_.MeanSquaredErrorLossRowwise()
        self._test_loss_function(f)

    def test_cross_entropy_loss_loss(self):
        f = sympy_.CrossEntropyLossRowwise()
        self._test_loss_function(f)

    def test_softmax_cross_entropy_loss(self):
        f = sympy_.SoftmaxCrossEntropyLossRowwise()
        self._test_loss_function_one_hot(f, colwise=False)

    def test_stable_softmax_cross_entropy_loss(self):
        f = sympy_.StableSoftmaxCrossEntropyLossRowwise()
        self._test_loss_function_one_hot(f, colwise=False)

    def test_logistic_cross_entropy_loss(self):
        f = sympy_.LogisticCrossEntropyLossRowwise()
        self._test_loss_function(f)

    def test_negative_log_likelihood_loss(self):
        f = sympy_.NegativeLogLikelihoodLossRowwise()
        self._test_loss_function(f)


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
        Y = np.array([
            [1, 2, 3],
            [7, 3, 4]
        ], dtype=float)

        T = np.array([
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=float)

        return Y, T

    def _test_loss_function(self, name, f_sympy, f_numpy, f_tensorflow, f_torch):
        Y, T = self.make_variables()

        x1 = f_sympy.value(to_sympy(Y), to_sympy(T))
        x2 = f_numpy.value(to_numpy(Y), to_numpy(T))
        x3 = tf_.squared_error_loss_colwise(to_tensorflow(Y), to_tensorflow(T))
        x4 = torch_.squared_error_loss_colwise(to_torch(Y), to_torch(T))
        self.check_numbers_equal(f'{name} values', [x1, x2, x3, x4])

        x1 = f_sympy.gradient(to_sympy(Y), to_sympy(T))
        x2 = f_numpy.gradient(to_numpy(Y), to_numpy(T))
        x3 = tf_.squared_error_loss_colwise_gradient(to_tensorflow(Y), to_tensorflow(T))
        x4 = torch_.squared_error_loss_colwise_gradient(to_torch(Y), to_torch(T))
        self.check_arrays_equal(f'{name} gradients', [x1, x2, x3, x4])

    def test_squared_error_loss_colwise(self):
        f_sympy = sympy_.SquaredErrorLossColwise()
        f_numpy = np_.SquaredErrorLossColwise()
        f_tensorflow = None
        f_torch = None
        self._test_loss_function('SquaredErrorLossColwise', f_sympy, f_numpy, f_tensorflow, f_torch)

    # def test_mean_squared_error_loss_colwise(self):
    #     f_sympy = sympy_.MeanSquaredErrorLossColwise()
    #     f_numpy = np_.MeanSquaredErrorLossColwise()
    #     f_tensorflow = None
    #     f_torch = None
    #     self._test_loss_function('MeanSquaredErrorLossColwise', f_sympy, f_numpy, f_tensorflow, f_torch)


if __name__ == '__main__':
    import unittest
    unittest.main()
