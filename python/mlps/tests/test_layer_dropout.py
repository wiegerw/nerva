#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase

from mlps.nerva_sympy.activation_functions import *
from mlps.nerva_sympy.matrix_operations import *
from mlps.tests.utilities import equal_matrices, matrix, squared_error


class TestDropoutLayers(TestCase):

    def test_linear_dropout_layer_colwise(self):
        D = 3
        N = 2
        K = 2
        loss = squared_error

        # variables
        x = matrix('x', D, N)
        y = matrix('y', K, N)
        w = matrix('w', K, D)
        b = matrix('b', K, 1)
        r = matrix('r', K, D)
        X = x
        W = w
        R = r

        # feedforward
        Y = hadamard(W, R) * X + column_repeat(b, N)

        # symbolic differentiation
        DY = substitute(diff(loss(y), y), (y, Y))

        # backpropagation
        DW = hadamard(DY * X.T, R)
        Db = rows_sum(DY)
        DX = hadamard(W, R).T * DY

        # test gradients
        DW1 = diff(loss(Y), w)
        Db1 = diff(loss(Y), b)
        DX1 = diff(loss(Y), x)

        self.assertTrue(equal_matrices(DW, DW1))
        self.assertTrue(equal_matrices(Db, Db1))
        self.assertTrue(equal_matrices(DX, DX1))

    def test_activation_dropout_layer_colwise(self):
        D = 3
        N = 2
        K = 2
        loss = squared_error
        act = Hyperbolic_tangent
        act_gradient = Hyperbolic_tangent_gradient

        # variables
        x = matrix('x', D, N)
        y = matrix('y', K, N)
        z = matrix('z', K, N)
        w = matrix('w', K, D)
        b = matrix('b', K, 1)
        r = matrix('r', K, D)
        X = x
        W = w
        R = r

        # feedforward
        Z = hadamard(W, R) * X + column_repeat(b, N)
        Y = act(Z)

        # symbolic differentiation
        DY = substitute(diff(loss(y), y), (y, Y))

        # backpropagation
        DZ = hadamard(DY, act_gradient(Z))
        DW = hadamard(DZ * X.T, R)
        Db = rows_sum(DZ)
        DX = hadamard(W, R).T * DZ

        # test gradients
        DZ1 = substitute(diff(loss(act(z)), z), (z, Z))
        DW1 = diff(loss(Y), w)
        Db1 = diff(loss(Y), b)
        DX1 = diff(loss(Y), x)

        self.assertTrue(equal_matrices(DZ, DZ1))
        self.assertTrue(equal_matrices(DW, DW1))
        self.assertTrue(equal_matrices(Db, Db1))
        self.assertTrue(equal_matrices(DX, DX1))

    def test_sigmoid_dropout_layer_colwise(self):
        D = 3
        N = 2
        K = 2
        loss = squared_error

        # variables
        x = matrix('x', D, N)
        y = matrix('y', K, N)
        z = matrix('z', K, N)
        w = matrix('w', K, D)
        b = matrix('b', K, 1)
        r = matrix('r', K, D)
        X = x
        W = w
        R = r

        # feedforward
        Z = hadamard(W, R) * X + column_repeat(b, N)
        Y = Sigmoid(Z)

        # symbolic differentiation
        DY = substitute(diff(loss(y), y), (y, Y))

        # backpropagation
        DZ = hadamard(DY, hadamard(Y, ones(K, N) - Y))
        DW = hadamard(DZ * X.T, R)
        Db = rows_sum(DZ)
        DX = hadamard(W, R).T * DZ

        # test gradients
        Y_z = Sigmoid(z)
        DZ1 = substitute(diff(loss(Y_z), z), (z, Z))
        DW1 = diff(loss(Y), w)
        Db1 = diff(loss(Y), b)
        DX1 = diff(loss(Y), x)

        self.assertTrue(equal_matrices(DZ, DZ1))
        self.assertTrue(equal_matrices(DW, DW1))
        self.assertTrue(equal_matrices(Db, Db1))
        self.assertTrue(equal_matrices(DX, DX1))

    def test_linear_dropout_layer_rowwise(self):
        D = 3
        N = 2
        K = 2
        loss = squared_error

        # variables
        x = matrix('x', N, D)
        y = matrix('y', N, K)
        w = matrix('w', K, D)
        b = matrix('b', 1, K)
        r = matrix('r', D, K)
        X = x
        W = w
        R = r

        # feedforward
        Y = X * hadamard(W.T, R) + row_repeat(b, N)

        # symbolic differentiation
        DY = substitute(diff(loss(y), y), (y, Y))

        # backpropagation
        DW = hadamard(DY.T * X, R.T)
        Db = columns_sum(DY)
        DX = DY * hadamard(W, R.T)

        # test gradients
        DW1 = diff(loss(Y), w)
        Db1 = diff(loss(Y), b)
        DX1 = diff(loss(Y), x)

        self.assertTrue(equal_matrices(DW, DW1))
        self.assertTrue(equal_matrices(Db, Db1))
        self.assertTrue(equal_matrices(DX, DX1))

    def test_activation_dropout_layer_rowwise(self):
        D = 3
        N = 2
        K = 2
        loss = squared_error
        act = Hyperbolic_tangent
        act_gradient = Hyperbolic_tangent_gradient

        # variables
        x = matrix('x', N, D)
        y = matrix('y', N, K)
        z = matrix('z', N, K)
        w = matrix('w', K, D)
        b = matrix('b', 1, K)
        r = matrix('r', D, K)
        X = x
        W = w
        R = r

        # feedforward
        Z = X * hadamard(W.T, R) + row_repeat(b, N)
        Y = act(Z)

        # symbolic differentiation
        DY = substitute(diff(loss(y), y), (y, Y))

        # backpropagation
        DZ = hadamard(DY, act_gradient(Z))
        DW = hadamard(DZ.T * X, R.T)
        Db = columns_sum(DZ)
        DX = DZ * hadamard(W, R.T)

        # test gradients
        DZ1 = substitute(diff(loss(act(z)), z), (z, Z))
        DW1 = diff(loss(Y), w)
        Db1 = diff(loss(Y), b)
        DX1 = diff(loss(Y), x)

        self.assertTrue(equal_matrices(DZ, DZ1))
        self.assertTrue(equal_matrices(DW, DW1))
        self.assertTrue(equal_matrices(Db, Db1))
        self.assertTrue(equal_matrices(DX, DX1))

    def test_sigmoid_dropout_layer_rowwise(self):
        D = 3
        N = 2
        K = 2
        loss = squared_error
        sigma = Sigmoid

        # variables
        x = matrix('x', N, D)
        y = matrix('y', N, K)
        z = matrix('z', N, K)
        w = matrix('w', K, D)
        b = matrix('b', 1, K)
        r = matrix('r', D, K)
        X = x
        W = w
        R = r

        # feedforward
        Z = X * hadamard(W.T, R) + row_repeat(b, N)
        Y = Sigmoid(Z)

        # symbolic differentiation
        DY = substitute(diff(loss(y), y), (y, Y))

        # backpropagation
        DZ = hadamard(DY, hadamard(Y, ones(N, K) - Y))
        DW = hadamard(DZ.T * X, R.T)
        Db = columns_sum(DZ)
        DX = DZ * hadamard(W, R.T)

        # test gradients
        Y_z = Sigmoid(z)
        DZ1 = substitute(diff(loss(Y_z), z), (z, Z))
        DW1 = diff(loss(Y), w)
        Db1 = diff(loss(Y), b)
        DX1 = diff(loss(Y), x)

        self.assertTrue(equal_matrices(DZ, DZ1))
        self.assertTrue(equal_matrices(DW, DW1))
        self.assertTrue(equal_matrices(Db, Db1))
        self.assertTrue(equal_matrices(DX, DX1))


if __name__ == '__main__':
    import unittest
    unittest.main()
