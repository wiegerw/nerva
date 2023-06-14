#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase

import numpy as np

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
        x1 = sympy_.Relu(to_sympy(x))
        x2 = np_.Relu(to_numpy(x))
        x3 = tf_.Relu(to_tensorflow(x))
        x4 = torch_.Relu(to_torch(x))
        self.check_arrays_equal('Relu', x1, x2, x3, x4)

        x1 = sympy_.Relu_gradient(to_sympy(x))
        x2 = np_.Relu_gradient(to_numpy(x))
        x3 = tf_.Relu_gradient(to_tensorflow(x))
        x4 = torch_.Relu_gradient(to_torch(x))
        self.check_arrays_equal('Relu_gradient', x1, x2, x3, x4)

    def test_leaky_relu(self):
        X, alpha, al, tl, ar, tr = self.make_variables()

        x = X
        x1 = sympy_.Leaky_relu(alpha)(to_sympy(x))
        x2 = np_.Leaky_relu(alpha)(to_numpy(x))
        x3 = tf_.Leaky_relu(alpha)(to_tensorflow(x))
        x4 = torch_.Leaky_relu(alpha)(to_torch(x))
        self.check_arrays_equal('Leaky_relu', x1, x2, x3, x4)

        x1 = sympy_.Leaky_relu_gradient(alpha)(to_sympy(x))
        x2 = np_.Leaky_relu_gradient(alpha)(to_numpy(x))
        x3 = tf_.Leaky_relu_gradient(alpha)(to_tensorflow(x))
        x4 = torch_.Leaky_relu_gradient(alpha)(to_torch(x))
        self.check_arrays_equal('Leaky_relu_gradient', x1, x2, x3, x4)

    def test_All_relu(self):
        X, alpha, al, tl, ar, tr = self.make_variables()

        x = X
        x1 = sympy_.All_relu(alpha)(to_sympy(x))
        x2 = np_.All_relu(alpha)(to_numpy(x))
        x3 = tf_.All_relu(alpha)(to_tensorflow(x))
        x4 = torch_.All_relu(alpha)(to_torch(x))
        self.check_arrays_equal('All_relu', x1, x2, x3, x4)

        x1 = sympy_.All_relu_gradient(alpha)(to_sympy(x))
        x2 = np_.All_relu_gradient(alpha)(to_numpy(x))
        x3 = tf_.All_relu_gradient(alpha)(to_tensorflow(x))
        x4 = torch_.All_relu_gradient(alpha)(to_torch(x))
        self.check_arrays_equal('All_relu_gradient', x1, x2, x3, x4)

    def test_Hyperbolic_tangent(self):
        X, alpha, al, tl, ar, tr = self.make_variables()

        x = X
        x1 = sympy_.Hyperbolic_tangent(to_sympy(x))
        x2 = np_.Hyperbolic_tangent(to_numpy(x))
        x3 = tf_.Hyperbolic_tangent(to_tensorflow(x))
        x4 = torch_.Hyperbolic_tangent(to_torch(x))
        self.check_arrays_equal('Hyperbolic_tangent', x1, x2, x3, x4)

        x1 = sympy_.Hyperbolic_tangent_gradient(to_sympy(x))
        x2 = np_.Hyperbolic_tangent_gradient(to_numpy(x))
        x3 = tf_.Hyperbolic_tangent_gradient(to_tensorflow(x))
        x4 = torch_.Hyperbolic_tangent_gradient(to_torch(x))
        self.check_arrays_equal('Hyperbolic_tangent_gradient', x1, x2, x3, x4)

    def test_Sigmoid(self):
        X, alpha, al, tl, ar, tr = self.make_variables()

        x = X
        x1 = sympy_.Sigmoid(to_sympy(x))
        x2 = np_.Sigmoid(to_numpy(x))
        x3 = tf_.Sigmoid(to_tensorflow(x))
        x4 = torch_.Sigmoid(to_torch(x))
        self.check_arrays_equal('Sigmoid', x1, x2, x3, x4)

        x1 = sympy_.Sigmoid_gradient(to_sympy(x))
        x2 = np_.Sigmoid_gradient(to_numpy(x))
        x3 = tf_.Sigmoid_gradient(to_tensorflow(x))
        x4 = torch_.Sigmoid_gradient(to_torch(x))
        self.check_arrays_equal('Sigmoid_gradient', x1, x2, x3, x4)

    def test_Srelu(self):
        X, alpha, al, tl, ar, tr = self.make_variables()

        x = X
        x1 = sympy_.Srelu(al, tl, ar, tr)(to_sympy(x))
        x2 = np_.Srelu(al, tl, ar, tr)(to_numpy(x))
        x3 = tf_.Srelu(al, tl, ar, tr)(to_tensorflow(x))
        x4 = torch_.Srelu(al, tl, ar, tr)(to_torch(x))
        self.check_arrays_equal('Srelu', x1, x2, x3, x4)

        x1 = sympy_.Srelu_gradient(al, tl, ar, tr)(to_sympy(x))
        x2 = np_.Srelu_gradient(al, tl, ar, tr)(to_numpy(x))
        x3 = tf_.Srelu_gradient(al, tl, ar, tr)(to_tensorflow(x))
        x4 = torch_.Srelu_gradient(al, tl, ar, tr)(to_torch(x))
        self.check_arrays_equal('Srelu_gradient', x1, x2, x3, x4)


if __name__ == '__main__':
    import unittest
    unittest.main()
