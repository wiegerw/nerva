#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase

import numpy as np

from symbolic.matrix_operations_sympy import substitute, elements_sum
from symbolic.utilities import to_numpy, to_sympy, to_torch, to_tensorflow, matrix, equal_matrices, instantiate, \
    instantiate_one_hot_colwise
import symbolic.loss_functions_numpy as np_
import symbolic.loss_functions_tensorflow as tf_
import symbolic.loss_functions_torch as torch_
import symbolic.loss_functions_sympy as sympy_

class TestLossFunctionVectorGradients(TestCase):
    def make_variables(self):
        K = 2
        y = matrix('y', K, 1)
        t = matrix('t', K, 1)
        return K, y, t

    def test_squared_error_loss(self):
        K, y, t = self.make_variables()
        loss = sympy_.squared_error_loss_colwise_vector(y, t)
        Dy1 = sympy_.squared_error_loss_colwise_gradient_vector(y, t)
        Dy2 = sympy_.diff(loss, y)
        self.assertTrue(equal_matrices(Dy1, Dy2))

    def test_mean_squared_error_loss(self):
        K, y, t = self.make_variables()
        loss = sympy_.mean_squared_error_loss_colwise_vector(y, t)
        Dy1 = sympy_.mean_squared_error_loss_colwise_gradient_vector(y, t)
        Dy2 = sympy_.diff(loss, y)
        self.assertTrue(equal_matrices(Dy1, Dy2))

    def test_cross_entropy_loss_loss(self):
        K, y, t = self.make_variables()
        loss = sympy_.cross_entropy_loss_colwise_vector(y, t)
        Dy1 = sympy_.cross_entropy_loss_colwise_gradient_vector(y, t)
        Dy2 = sympy_.diff(loss, y)
        self.assertTrue(equal_matrices(Dy1, Dy2))

    def test_softmax_cross_entropy_loss(self):
        K, y, t = self.make_variables()
        loss = sympy_.softmax_cross_entropy_loss_colwise_vector(y, t)
        Dy1 = sympy_.softmax_cross_entropy_loss_colwise_gradient_vector(y, t)
        Dy2 = sympy_.diff(loss, y)
        self.assertTrue(equal_matrices(Dy1, Dy2))

        # test with a one-hot encoded vector t0
        t0 = instantiate_one_hot_colwise(t)
        assert elements_sum(t0) == 1
        Dy1_a = substitute(sympy_.softmax_cross_entropy_one_hot_loss_colwise_gradient_vector(y, t), (t, t0))
        Dy2_a = substitute(Dy2, (t, t0))
        self.assertTrue(equal_matrices(Dy1_a, Dy2_a))

    def test_stable_softmax_cross_entropy_loss(self):
        K, y, t = self.make_variables()
        loss = sympy_.stable_softmax_cross_entropy_loss_colwise_vector(y, t)
        Dy1 = sympy_.stable_softmax_cross_entropy_loss_colwise_gradient_vector(y, t)
        Dy2 = sympy_.diff(loss, y)
        self.assertTrue(equal_matrices(Dy1, Dy2))

        # test with a one-hot encoded vector t0
        t0 = instantiate_one_hot_colwise(t)
        assert elements_sum(t0) == 1
        Dy1_a = substitute(sympy_.stable_softmax_cross_entropy_one_hot_loss_colwise_gradient_vector(y, t), (t, t0))
        Dy2_a = substitute(Dy2, (t, t0))
        self.assertTrue(equal_matrices(Dy1_a, Dy2_a))

    def test_logistic_cross_entropy_loss(self):
        K, y, t = self.make_variables()
        loss = sympy_.logistic_cross_entropy_loss_colwise_vector(y, t)
        Dy1 = sympy_.logistic_cross_entropy_loss_colwise_gradient_vector(y, t)
        Dy2 = sympy_.diff(loss, y)
        self.assertTrue(equal_matrices(Dy1, Dy2))

    def test_negative_log_likelihood_loss(self):
        K, y, t = self.make_variables()
        loss = sympy_.negative_log_likelihood_loss_colwise_vector(y, t)
        Dy1 = sympy_.negative_log_likelihood_loss_colwise_gradient_vector(y, t)
        Dy2 = sympy_.diff(loss, y)
        self.assertTrue(equal_matrices(Dy1, Dy2))


class TestLossFunctionGradients(TestCase):
    def make_variables(self):
        K = 3
        N = 2
        Y = matrix('Y', K, N)
        T = matrix('T', K, N)
        return K, N, Y, T

    def test_squared_error_loss(self):
        K, N, Y, T = self.make_variables()
        loss = sympy_.squared_error_loss(Y, T)
        DY1 = sympy_.squared_error_loss_colwise_gradient(Y, T)
        DY2 = sympy_.diff(loss, Y)
        self.assertTrue(equal_matrices(DY1, DY2))

    def test_mean_squared_error_loss(self):
        K, N, Y, T = self.make_variables()
        loss = sympy_.mean_squared_error_loss(Y, T)
        DY1 = sympy_.mean_squared_error_loss_colwise_gradient(Y, T)
        DY2 = sympy_.diff(loss, Y)
        self.assertTrue(equal_matrices(DY1, DY2))

    def test_cross_entropy_loss_loss(self):
        K, N, Y, T = self.make_variables()
        loss = sympy_.cross_entropy_loss(Y, T)
        DY1 = sympy_.cross_entropy_loss_colwise_gradient(Y, T)
        DY2 = sympy_.diff(loss, Y)
        self.assertTrue(equal_matrices(DY1, DY2))

    def test_softmax_cross_entropy_loss(self):
        K, N, Y, T = self.make_variables()
        loss = sympy_.softmax_cross_entropy_loss(Y, T)
        DY1 = sympy_.softmax_cross_entropy_loss_colwise_gradient(Y, T)
        DY2 = sympy_.diff(loss, Y)
        self.assertTrue(equal_matrices(DY1, DY2))

        # test with a one-hot encoded matrix T0
        T0 = instantiate_one_hot_colwise(T)
        DY1_a = substitute(sympy_.softmax_cross_entropy_one_hot_loss_colwise_gradient_vector(Y, T), (T, T0))
        DY2_a = substitute(DY2, (T, T0))
        self.assertTrue(equal_matrices(DY1_a, DY2_a))

    def test_stable_softmax_cross_entropy_loss(self):
        K, N, Y, T = self.make_variables()
        loss = sympy_.stable_softmax_cross_entropy_loss(Y, T)
        DY1 = sympy_.stable_softmax_cross_entropy_loss_colwise_gradient(Y, T)
        DY2 = sympy_.diff(loss, Y)
        self.assertTrue(equal_matrices(DY1, DY2))

        # test with a one-hot encoded matrix T0
        T0 = instantiate_one_hot_colwise(T)
        DY1_a = substitute(sympy_.stable_softmax_cross_entropy_one_hot_loss_colwise_gradient_vector(Y, T), (T, T0))
        DY2_a = substitute(DY2, (T, T0))
        self.assertTrue(equal_matrices(DY1_a, DY2_a))

    def test_logistic_cross_entropy_loss(self):
        K, N, Y, T = self.make_variables()
        loss = sympy_.logistic_cross_entropy_loss(Y, T)
        DY1 = sympy_.logistic_cross_entropy_loss_colwise_gradient(Y, T)
        DY2 = sympy_.diff(loss, Y)
        self.assertTrue(equal_matrices(DY1, DY2))

    def test_negative_log_likelihood_loss(self):
        K, N, Y, T = self.make_variables()
        loss = sympy_.negative_log_likelihood_loss(Y, T)
        DY1 = sympy_.negative_log_likelihood_loss_colwise_gradient(Y, T)
        DY2 = sympy_.diff(loss, Y)
        self.assertTrue(equal_matrices(DY1, DY2))


class TestLossFunctions(TestCase):
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
        self.assertTrue(np.allclose(x1, x2, atol=1e-5))
        self.assertTrue(np.allclose(x1, x3, atol=1e-5))
        self.assertTrue(np.allclose(x1, x4, atol=1e-5))

    def check_numbers_equal(self, operation, x1, x2, x3, x4):
        print(f'--- {operation} ---')
        print(x1, x1.__class__)
        print(x2, x2.__class__)
        print(x3, x3.__class__)
        print(x4, x4.__class__)
        self.assertAlmostEqual(x1, x2, delta=1e-5)
        self.assertAlmostEqual(x1, x3, delta=1e-5)
        self.assertAlmostEqual(x1, x4, delta=1e-5)

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

    def test_squared_error_loss(self):
        Y, T = self.make_variables()

        x1 = sympy_.squared_error_loss(to_sympy(Y), to_sympy(T))
        x2 = np_.squared_error_loss(to_numpy(Y), to_numpy(T))
        x3 = tf_.squared_error_loss(to_tensorflow(Y), to_tensorflow(T))
        x4 = torch_.squared_error_loss(to_torch(Y), to_torch(T))
        self.check_numbers_equal('squared_error_loss', x1, x2, x3, x4)

        x1 = sympy_.squared_error_loss_colwise_gradient(to_sympy(Y), to_sympy(T))
        x2 = np_.squared_error_loss_colwise_gradient(to_numpy(Y), to_numpy(T))
        x3 = tf_.squared_error_loss_colwise_gradient(to_tensorflow(Y), to_tensorflow(T))
        x4 = torch_.squared_error_loss_colwise_gradient(to_torch(Y), to_torch(T))
        self.check_arrays_equal('squared_error_loss_colwise_gradient', x1, x2, x3, x4)

    def test_mean_squared_error_loss(self):
        Y, T = self.make_variables()

        x1 = sympy_.mean_squared_error_loss(to_sympy(Y), to_sympy(T))
        x2 = np_.mean_squared_error_loss(to_numpy(Y), to_numpy(T))
        x3 = tf_.mean_squared_error_loss(to_tensorflow(Y), to_tensorflow(T))
        x4 = torch_.mean_squared_error_loss(to_torch(Y), to_torch(T))
        self.check_numbers_equal('mean_squared_error_loss', x1, x2, x3, x4)

        x1 = sympy_.mean_squared_error_loss_colwise_gradient(to_sympy(Y), to_sympy(T))
        x2 = np_.mean_squared_error_loss_colwise_gradient(to_numpy(Y), to_numpy(T))
        x3 = tf_.mean_squared_error_loss_colwise_gradient(to_tensorflow(Y), to_tensorflow(T))
        x4 = torch_.mean_squared_error_loss_colwise_gradient(to_torch(Y), to_torch(T))
        self.check_arrays_equal('mean_squared_error_loss_colwise_gradient', x1, x2, x3, x4)


if __name__ == '__main__':
    import unittest
    unittest.main()
