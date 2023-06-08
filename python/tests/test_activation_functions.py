#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase

import numpy as np
import sympy as sp
import torch
import tensorflow as tf

from symbolic.utilities import to_numpy, to_sympy, to_torch, to_tensorflow
import symbolic.activation_functions_numpy as np_
import symbolic.activation_functions_tensorflow as tf_
import symbolic.activation_functions_torch as torch_
import symbolic.activation_functions_sympy as sympy_


class TestActivationFunctions(TestCase):
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

    def make_variables(self):
        X = np.array([
            [1, 2, 3],
            [7, 3, 4]
        ], dtype=float)

        alpha = 0.25
        al = 0.2
        tl = 0.1
        ar = 0.7
        tr = 0.3

        return X, alpha, al, tl, ar, tr

    def test_relu(self):
        X, alpha, al, tl, ar, tr = self.make_variables()

        x = X
        x1 = sympy_.relu(to_sympy(x))
        x2 = np_.relu(to_numpy(x))
        x3 = tf_.relu(to_tensorflow(x))
        x4 = torch_.relu(to_torch(x))
        self.check_arrays_equal('relu', x1, x2, x3, x4)

        x1 = sympy_.relu_prime(to_sympy(x))
        x2 = np_.relu_prime(to_numpy(x))
        x3 = tf_.relu_prime(to_tensorflow(x))
        x4 = torch_.relu_prime(to_torch(x))
        self.check_arrays_equal('relu_prime', x1, x2, x3, x4)

    def test_leaky_relu(self):
        X, alpha, al, tl, ar, tr = self.make_variables()

        x = X
        x1 = sympy_.leaky_relu(alpha)(to_sympy(x))
        x2 = np_.leaky_relu(alpha)(to_numpy(x))
        x3 = tf_.leaky_relu(alpha)(to_tensorflow(x))
        x4 = torch_.leaky_relu(alpha)(to_torch(x))
        self.check_arrays_equal('leaky_relu', x1, x2, x3, x4)

        x1 = sympy_.leaky_relu_prime(alpha)(to_sympy(x))
        x2 = np_.leaky_relu_prime(alpha)(to_numpy(x))
        x3 = tf_.leaky_relu_prime(alpha)(to_tensorflow(x))
        x4 = torch_.leaky_relu_prime(alpha)(to_torch(x))
        self.check_arrays_equal('leaky_relu_prime', x1, x2, x3, x4)

    def test_all_relu(self):
        X, alpha, al, tl, ar, tr = self.make_variables()

        x = X
        x1 = sympy_.all_relu(alpha)(to_sympy(x))
        x2 = np_.all_relu(alpha)(to_numpy(x))
        x3 = tf_.all_relu(alpha)(to_tensorflow(x))
        x4 = torch_.all_relu(alpha)(to_torch(x))
        self.check_arrays_equal('all_relu', x1, x2, x3, x4)

        x1 = sympy_.all_relu_prime(alpha)(to_sympy(x))
        x2 = np_.all_relu_prime(alpha)(to_numpy(x))
        x3 = tf_.all_relu_prime(alpha)(to_tensorflow(x))
        x4 = torch_.all_relu_prime(alpha)(to_torch(x))
        self.check_arrays_equal('all_relu_prime', x1, x2, x3, x4)

    def test_hyperbolic_tangent(self):
        X, alpha, al, tl, ar, tr = self.make_variables()

        x = X
        x1 = sympy_.hyperbolic_tangent(to_sympy(x))
        x2 = np_.hyperbolic_tangent(to_numpy(x))
        x3 = tf_.hyperbolic_tangent(to_tensorflow(x))
        x4 = torch_.hyperbolic_tangent(to_torch(x))
        self.check_arrays_equal('hyperbolic_tangent', x1, x2, x3, x4)

        x1 = sympy_.hyperbolic_tangent_prime(to_sympy(x))
        x2 = np_.hyperbolic_tangent_prime(to_numpy(x))
        x3 = tf_.hyperbolic_tangent_prime(to_tensorflow(x))
        x4 = torch_.hyperbolic_tangent_prime(to_torch(x))
        self.check_arrays_equal('hyperbolic_tangent_prime', x1, x2, x3, x4)

    def test_sigmoid(self):
        X, alpha, al, tl, ar, tr = self.make_variables()

        x = X
        x1 = sympy_.sigmoid(to_sympy(x))
        x2 = np_.sigmoid(to_numpy(x))
        x3 = tf_.sigmoid(to_tensorflow(x))
        x4 = torch_.sigmoid(to_torch(x))
        self.check_arrays_equal('sigmoid', x1, x2, x3, x4)

        x1 = sympy_.sigmoid_prime(to_sympy(x))
        x2 = np_.sigmoid_prime(to_numpy(x))
        x3 = tf_.sigmoid_prime(to_tensorflow(x))
        x4 = torch_.sigmoid_prime(to_torch(x))
        self.check_arrays_equal('sigmoid_prime', x1, x2, x3, x4)

    def test_srelu(self):
        X, alpha, al, tl, ar, tr = self.make_variables()

        x = X
        x1 = sympy_.srelu(al, tl, ar, tr)(to_sympy(x))
        x2 = np_.srelu(al, tl, ar, tr)(to_numpy(x))
        x3 = tf_.srelu(al, tl, ar, tr)(to_tensorflow(x))
        x4 = torch_.srelu(al, tl, ar, tr)(to_torch(x))
        self.check_arrays_equal('srelu', x1, x2, x3, x4)

        x1 = sympy_.srelu_prime(al, tl, ar, tr)(to_sympy(x))
        x2 = np_.srelu_prime(al, tl, ar, tr)(to_numpy(x))
        x3 = tf_.srelu_prime(al, tl, ar, tr)(to_tensorflow(x))
        x4 = torch_.srelu_prime(al, tl, ar, tr)(to_torch(x))
        self.check_arrays_equal('srelu_prime', x1, x2, x3, x4)


if __name__ == '__main__':
    import unittest
    unittest.main()
