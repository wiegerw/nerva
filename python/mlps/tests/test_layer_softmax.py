#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase

from mlps.nerva_sympy.matrix_operations import *
from mlps.nerva_sympy.softmax_functions import *
from mlps.tests.utilities import equal_matrices, matrix, squared_error


class TestSoftmaxLayers(TestCase):

    def test_softmax_layer_colwise(self):
        D = 3
        K = 2
        N = 2
        loss = squared_error

        # variables
        x = matrix('x', D, N)
        y = matrix('y', K, N)
        z = matrix('z', K, N)
        w = matrix('w', K, D)
        b = matrix('b', K, 1)
        X = x
        W = w

        # feedforward
        Z = W * X + column_repeat(b, N)
        Y = softmax_colwise(Z)

        # symbolic differentiation
        DY = substitute(gradient(loss(y), y), (y, Y))

        # backpropagation
        DZ = hadamard(Y, DY - row_repeat(diag(Y.T * DY).T, K))
        DW = DZ * X.T
        Db = rows_sum(DZ)
        DX = W.T * DZ

        # test gradients
        DW1 = gradient(loss(Y), w)
        Db1 = gradient(loss(Y), b)
        DX1 = gradient(loss(Y), x)
        self.assertTrue(equal_matrices(DW, DW1))
        self.assertTrue(equal_matrices(Db, Db1))
        self.assertTrue(equal_matrices(DX, DX1))

        # test DZ using Z = z
        Z = z
        Y = softmax_colwise(Z)
        DY = substitute(gradient(loss(y), y), (y, Y))
        DZ = hadamard(Y, DY - row_repeat(diag(Y.T * DY).T, K))
        DZ1 = gradient(loss(Y), z)
        self.assertTrue(equal_matrices(DZ, DZ1))

    def test_log_softmax_layer_colwise(self):
        D = 2
        K = 2
        N = 2
        loss = squared_error

        # variables
        x = matrix('x', D, N)
        y = matrix('y', K, N)
        z = matrix('z', K, N)
        w = matrix('w', K, D)
        b = matrix('b', K, 1)
        X = x
        W = w

        # feedforward
        Z = W * X + column_repeat(b, N)
        Y = log_softmax_colwise(Z)

        # symbolic differentiation
        DY = substitute(gradient(loss(y), y), (y, Y))

        # backpropagation
        DZ = DY - hadamard(softmax_colwise(Z), row_repeat(columns_sum(DY), K))
        DW = DZ * X.T
        Db = rows_sum(DZ)
        DX = W.T * DZ

        # test gradients
        DW1 = gradient(loss(Y), w)
        Db1 = gradient(loss(Y), b)
        DX1 = gradient(loss(Y), x)
        # N.B. These tests take a long time, and are duplicates of the ones in test_log_softmax_layer_rowwise
        # self.assertTrue(equal_matrices(DW, DW1))
        # self.assertTrue(equal_matrices(Db, Db1))
        # self.assertTrue(equal_matrices(DX, DX1))

        # test DZ using Z = z
        Z = z
        Y = log_softmax_colwise(Z)
        DY = substitute(gradient(loss(y), y), (y, Y))
        DZ = DY - hadamard(softmax_colwise(Z), row_repeat(columns_sum(DY), K))
        DZ1 = gradient(loss(Y), z)
        self.assertTrue(equal_matrices(DZ, DZ1))

    def test_softmax_layer_rowwise(self):
        D = 3
        K = 2
        N = 2
        loss = squared_error

        # variables
        x = matrix('x', N, D)
        y = matrix('y', N, K)
        z = matrix('z', N, K)
        w = matrix('w', K, D)
        b = matrix('b', 1, K)
        X = x
        W = w

        # feedforward
        Z = X * W.T + row_repeat(b, N)
        Y = softmax_rowwise(Z)

        # symbolic differentiation
        DY = substitute(gradient(loss(y), y), (y, Y))

        # backpropagation
        DZ = hadamard(Y, DY - column_repeat(diag(DY * Y.T), N))
        DW = DZ.T * X
        Db = columns_sum(DZ)
        DX = DZ * W

        # test gradients
        DW1 = gradient(loss(Y), w)
        Db1 = gradient(loss(Y), b)
        DX1 = gradient(loss(Y), x)
        self.assertTrue(equal_matrices(DW, DW1))
        self.assertTrue(equal_matrices(Db, Db1))
        self.assertTrue(equal_matrices(DX, DX1))

        # test DZ using Z = z
        Z = z
        Y = softmax_rowwise(Z)
        DY = substitute(gradient(loss(y), y), (y, Y))
        DZ = hadamard(Y, DY - column_repeat(diag(DY * Y.T), N))
        DZ1 = gradient(loss(Y), z)
        self.assertTrue(equal_matrices(DZ, DZ1))

    def test_log_softmax_layer_rowwise(self):
        D = 3
        K = 2
        N = 2
        loss = elements_sum  # N.B. In this case squared_error seems too complicated

        # variables
        x = matrix('x', N, D)
        y = matrix('y', N, K)
        z = matrix('z', N, K)
        w = matrix('w', K, D)
        b = matrix('b', 1, K)
        X = x
        W = w

        # feedforward
        Z = X * W.T + row_repeat(b, N)
        Y = log_softmax_rowwise(Z)

        # symbolic differentiation
        DY = substitute(gradient(loss(y), y), (y, Y))

        # backpropagation
        DZ = DY - hadamard(softmax_rowwise(Z), column_repeat(rows_sum(DY), N))
        DW = DZ.T * X
        Db = columns_sum(DZ)
        DX = DZ * W

        # test gradients
        DW1 = gradient(loss(Y), w)
        Db1 = gradient(loss(Y), b)
        DX1 = gradient(loss(Y), x)
        self.assertTrue(equal_matrices(DW, DW1, True))
        self.assertTrue(equal_matrices(Db, Db1, True))
        self.assertTrue(equal_matrices(DX, DX1))

        # test DZ using Z = z
        Z = z
        Y = log_softmax_rowwise(Z)
        DY = substitute(gradient(loss(y), y), (y, Y))
        DZ = DY - hadamard(softmax_rowwise(Z), column_repeat(rows_sum(DY), N))
        DZ1 = gradient(loss(Y), z)
        self.assertTrue(equal_matrices(DZ, DZ1))


if __name__ == '__main__':
    import unittest
    unittest.main()
