#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase
import numpy as np
import nerva_numpy.activation_functions as np_
import nerva_tensorflow.activation_functions as tf_
import nerva_torch.activation_functions as torch_
import nerva_sympy.activation_functions as sympy_
import nerva_jax.activation_functions as jnp_
import nervalibcolwise as eigen_
from tests.test_utilities import to_numpy, to_sympy, to_torch, to_tensorflow, to_jax, to_eigen


class TestActivationFunctions(TestCase):
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
        X = np.array([
            [1, 2, 3],
            [7, 3, 4]
        ], dtype=np.float32)

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
        x5 = jnp_.Relu(to_jax(x))
        x6 = eigen_.Relu(to_eigen(x))
        self.check_arrays_equal('Relu', [x1, x2, x3, x4, x5, x6])

        x1 = sympy_.Relu_gradient(to_sympy(x))
        x2 = np_.Relu_gradient(to_numpy(x))
        x3 = tf_.Relu_gradient(to_tensorflow(x))
        x4 = torch_.Relu_gradient(to_torch(x))
        x5 = jnp_.Relu_gradient(to_jax(x))
        x6 = eigen_.Relu_gradient(to_eigen(x))
        self.check_arrays_equal('Relu_gradient', [x1, x2, x3, x4, x5, x6])

    def test_leaky_relu(self):
        X, alpha, al, tl, ar, tr = self.make_variables()

        x = X
        x1 = sympy_.Leaky_relu(alpha)(to_sympy(x))
        x2 = np_.Leaky_relu(alpha)(to_numpy(x))
        x3 = tf_.Leaky_relu(alpha)(to_tensorflow(x))
        x4 = torch_.Leaky_relu(alpha)(to_torch(x))
        x5 = jnp_.Leaky_relu(alpha)(to_jax(x))
        x6 = eigen_.Leaky_relu(alpha)(to_eigen(x))
        self.check_arrays_equal('Leaky_relu', [x1, x2, x3, x4, x5, x6])

        x1 = sympy_.Leaky_relu_gradient(alpha)(to_sympy(x))
        x2 = np_.Leaky_relu_gradient(alpha)(to_numpy(x))
        x3 = tf_.Leaky_relu_gradient(alpha)(to_tensorflow(x))
        x4 = torch_.Leaky_relu_gradient(alpha)(to_torch(x))
        x5 = jnp_.Leaky_relu_gradient(alpha)(to_jax(x))
        x6 = eigen_.Leaky_relu_gradient(alpha)(to_eigen(x))
        self.check_arrays_equal('Leaky_relu_gradient', [x1, x2, x3, x4, x5, x6])

    def test_All_relu(self):
        X, alpha, al, tl, ar, tr = self.make_variables()

        x = X
        x1 = sympy_.All_relu(alpha)(to_sympy(x))
        x2 = np_.All_relu(alpha)(to_numpy(x))
        x3 = tf_.All_relu(alpha)(to_tensorflow(x))
        x4 = torch_.All_relu(alpha)(to_torch(x))
        x5 = jnp_.All_relu(alpha)(to_jax(x))
        x6 = eigen_.All_relu(alpha)(to_eigen(x))
        self.check_arrays_equal('All_relu', [x1, x2, x3, x4, x5, x6])

        x1 = sympy_.All_relu_gradient(alpha)(to_sympy(x))
        x2 = np_.All_relu_gradient(alpha)(to_numpy(x))
        x3 = tf_.All_relu_gradient(alpha)(to_tensorflow(x))
        x4 = torch_.All_relu_gradient(alpha)(to_torch(x))
        x5 = jnp_.All_relu_gradient(alpha)(to_jax(x))
        x6 = eigen_.All_relu_gradient(alpha)(to_eigen(x))
        self.check_arrays_equal('All_relu_gradient', [x1, x2, x3, x4, x5, x6])

    def test_Hyperbolic_tangent(self):
        X, alpha, al, tl, ar, tr = self.make_variables()

        x = X
        x1 = sympy_.Hyperbolic_tangent(to_sympy(x))
        x2 = np_.Hyperbolic_tangent(to_numpy(x))
        x3 = tf_.Hyperbolic_tangent(to_tensorflow(x))
        x4 = torch_.Hyperbolic_tangent(to_torch(x))
        x5 = jnp_.Hyperbolic_tangent(to_jax(x))
        x6 = eigen_.Hyperbolic_tangent(to_eigen(x))
        self.check_arrays_equal('Hyperbolic_tangent', [x1, x2, x3, x4, x5, x6])

        x1 = sympy_.Hyperbolic_tangent_gradient(to_sympy(x))
        x2 = np_.Hyperbolic_tangent_gradient(to_numpy(x))
        x3 = tf_.Hyperbolic_tangent_gradient(to_tensorflow(x))
        x4 = torch_.Hyperbolic_tangent_gradient(to_torch(x))
        x5 = jnp_.Hyperbolic_tangent_gradient(to_jax(x))
        x6 = eigen_.Hyperbolic_tangent_gradient(to_eigen(x))
        self.check_arrays_equal('Hyperbolic_tangent_gradient', [x1, x2, x3, x4, x5, x6])

    def test_Sigmoid(self):
        X, alpha, al, tl, ar, tr = self.make_variables()

        x = X
        x1 = sympy_.Sigmoid(to_sympy(x))
        x2 = np_.Sigmoid(to_numpy(x))
        x3 = tf_.Sigmoid(to_tensorflow(x))
        x4 = torch_.Sigmoid(to_torch(x))
        x5 = jnp_.Sigmoid(to_jax(x))
        x6 = eigen_.Sigmoid(to_eigen(x))
        self.check_arrays_equal('Sigmoid', [x1, x2, x3, x4, x5, x6])

        x1 = sympy_.Sigmoid_gradient(to_sympy(x))
        x2 = np_.Sigmoid_gradient(to_numpy(x))
        x3 = tf_.Sigmoid_gradient(to_tensorflow(x))
        x4 = torch_.Sigmoid_gradient(to_torch(x))
        x5 = jnp_.Sigmoid_gradient(to_jax(x))
        x6 = eigen_.Sigmoid_gradient(to_eigen(x))
        self.check_arrays_equal('Sigmoid_gradient', [x1, x2, x3, x4, x5, x6])

    def test_Srelu(self):
        X, alpha, al, tl, ar, tr = self.make_variables()

        x = X
        x1 = sympy_.Srelu(al, tl, ar, tr)(to_sympy(x))
        x2 = np_.Srelu(al, tl, ar, tr)(to_numpy(x))
        x3 = tf_.Srelu(al, tl, ar, tr)(to_tensorflow(x))
        x4 = torch_.Srelu(al, tl, ar, tr)(to_torch(x))
        x5 = jnp_.Srelu(al, tl, ar, tr)(to_jax(x))
        x6 = eigen_.Srelu(al, tl, ar, tr)(to_eigen(x))
        self.check_arrays_equal('Srelu', [x1, x2, x3, x4, x5, x6])

        x1 = sympy_.Srelu_gradient(al, tl, ar, tr)(to_sympy(x))
        x2 = np_.Srelu_gradient(al, tl, ar, tr)(to_numpy(x))
        x3 = tf_.Srelu_gradient(al, tl, ar, tr)(to_tensorflow(x))
        x4 = torch_.Srelu_gradient(al, tl, ar, tr)(to_torch(x))
        x5 = jnp_.Srelu_gradient(al, tl, ar, tr)(to_jax(x))
        x6 = eigen_.Srelu_gradient(al, tl, ar, tr)(to_eigen(x))
        self.check_arrays_equal('Srelu_gradient', [x1, x2, x3, x4, x5, x6])


if __name__ == '__main__':
    import unittest
    unittest.main()
