#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase
from symbolic.activation_functions_sympy import *
from symbolic.matrix_operations_sympy import *
from symbolic.utilities import *
from symbolic.utilities import squared_error


class TestLinearLayers(TestCase):

    def test_linear_layer_colwise(self):
        D = 3
        K = 2
        N = 2
        loss = squared_error

        # variables
        x = matrix('x', D, N)
        y = matrix('y', K, N)
        w = matrix('w', K, D)
        b = matrix('b', K, 1)

        # feedforward
        X = x
        W = w
        Y = W * X + column_repeat(b, N)

        # backpropagation
        DY = substitute(diff(loss(y), y), (y, Y))
        DW = DY * X.T
        Db = rows_sum(DY)
        DX = W.T * DY

        # test gradients
        DW1 = diff(loss(Y), w)
        Db1 = diff(loss(Y), b)
        DX1 = diff(loss(Y), x)

        self.assertTrue(equal_matrices(DW, DW1))
        self.assertTrue(equal_matrices(Db, Db1))
        self.assertTrue(equal_matrices(DX, DX1))

    def test_activation_layer_colwise(self):
        D = 3
        K = 2
        N = 2
        loss = squared_error
        act = hyperbolic_tangent
        act_gradient = hyperbolic_tangent_gradient

        # variables
        x = matrix('x', D, N)
        y = matrix('y', K, N)
        z = matrix('z', K, N)
        w = matrix('w', K, D)
        b = matrix('b', K, 1)

        # feedforward
        X = x
        W = w
        Z = W * X + column_repeat(b, N)
        Y = act(Z)

        # backpropagation
        DY = substitute(diff(loss(y), y), (y, Y))
        DZ = hadamard(DY, act_gradient(Z))
        DW = DZ * X.T
        Db = rows_sum(DZ)
        DX = W.T * DZ

        # test gradients
        DZ1 = substitute(diff(loss(act(z)), z), (z, Z))
        DW1 = diff(loss(Y), w)
        Db1 = diff(loss(Y), b)
        DX1 = diff(loss(Y), x)

        self.assertTrue(equal_matrices(DZ, DZ1))
        self.assertTrue(equal_matrices(DW, DW1))
        self.assertTrue(equal_matrices(Db, Db1))
        self.assertTrue(equal_matrices(DX, DX1))

    def test_sigmoid_layer_colwise(self):
        D = 3
        K = 2
        N = 2
        loss = squared_error
        sigma = sigmoid

        # variables
        x = matrix('x', D, N)
        y = matrix('y', K, N)
        z = matrix('z', K, N)
        w = matrix('w', K, D)
        b = matrix('b', K, 1)

        # feedforward
        X = x
        W = w
        Z = W * X + column_repeat(b, N)
        Y = sigmoid(Z)

        # backpropagation
        DY = substitute(diff(loss(y), y), (y, Y))
        DZ = hadamard(DY, hadamard(Y, ones(K, N) - Y))
        DW = DZ * X.T
        Db = rows_sum(DZ)
        DX = W.T * DZ

        # test gradients
        Y_z = sigmoid(z)
        DZ1 = substitute(diff(loss(Y_z), z), (z, Z))
        DW1 = diff(loss(Y), w)
        Db1 = diff(loss(Y), b)
        DX1 = diff(loss(Y), x)

        self.assertTrue(equal_matrices(DZ, DZ1))
        self.assertTrue(equal_matrices(DW, DW1))
        self.assertTrue(equal_matrices(Db, Db1))
        self.assertTrue(equal_matrices(DX, DX1))

    def test_linear_layer_rowwise(self):
        D = 3
        K = 2
        N = 2
        loss = squared_error

        # variables
        x = matrix('x', N, D)
        y = matrix('y', N, K)
        w = matrix('w', K, D)
        b = matrix('b', 1, K)

        # feedforward
        X = x
        W = w
        Y = X * W.T + row_repeat(b, N)

        # backpropagation
        DY = substitute(diff(loss(y), y), (y, Y))
        DW = DY.T * X
        Db = columns_sum(DY)
        DX = DY * W

        # test gradients
        DW1 = diff(loss(Y), w)
        Db1 = diff(loss(Y), b)
        DX1 = diff(loss(Y), x)

        self.assertTrue(equal_matrices(DW, DW1))
        self.assertTrue(equal_matrices(Db, Db1))
        self.assertTrue(equal_matrices(DX, DX1))

    def test_activation_layer_rowwise(self):
        D = 3
        K = 2
        N = 2
        loss = squared_error
        act = hyperbolic_tangent
        act_gradient = hyperbolic_tangent_gradient

        # variables
        x = matrix('x', N, D)
        y = matrix('y', N, K)
        z = matrix('z', N, K)
        w = matrix('w', K, D)
        b = matrix('b', 1, K)

        # feedforward
        X = x
        W = w
        Z = X * W.T + row_repeat(b, N)
        Y = act(Z)

        # backpropagation
        DY = substitute(diff(loss(y), y), (y, Y))
        DZ = hadamard(DY, act_gradient(Z))
        DW = DZ.T * X
        Db = columns_sum(DZ)
        DX = DZ * W

        # test gradients
        DZ1 = substitute(diff(loss(act(z)), z), (z, Z))
        DW1 = diff(loss(Y), w)
        Db1 = diff(loss(Y), b)
        DX1 = diff(loss(Y), x)

        self.assertTrue(equal_matrices(DZ, DZ1))
        self.assertTrue(equal_matrices(DW, DW1))
        self.assertTrue(equal_matrices(Db, Db1))
        self.assertTrue(equal_matrices(DX, DX1))

    def test_sigmoid_layer_rowwise(self):
        D = 3
        K = 2
        N = 2
        loss = squared_error
        sigma = sigmoid

        # variables
        x = matrix('x', N, D)
        y = matrix('y', N, K)
        z = matrix('z', N, K)
        w = matrix('w', K, D)
        b = matrix('b', 1, K)

        # feedforward
        X = x
        W = w
        Z = X * W.T + row_repeat(b, N)
        Y = sigmoid(Z)

        # backpropagation
        DY = substitute(diff(loss(y), y), (y, Y))
        DZ = hadamard(DY, hadamard(Y, ones(N, K) - Y))
        DW = DZ.T * X
        Db = columns_sum(DZ)
        DX = DZ * W

        # test gradients
        Y_z = sigmoid(z)
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
