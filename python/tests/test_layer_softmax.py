#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase
from symbolic.loss_functions import *
from symbolic.matrix_operations_sympy import *
from symbolic.softmax import *
from symbolic.utilities import *


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

        # feedforward
        X = x
        W = w
        Z = W * X + column_repeat(b, N)
        Y = softmax_colwise(Z)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DZ = hadamard(Y, DY - row_repeat(diag(Y.T * DY).T, K))
        DW = DZ * X.T
        Db = rows_sum(DZ)
        DX = W.T * DZ

        # test gradients
        DW1 = diff(loss(Y), w)
        Db1 = diff(loss(Y), b)
        DX1 = diff(loss(Y), x)
        self.assertTrue(equal_matrices(DW, DW1))
        self.assertTrue(equal_matrices(Db, Db1))
        self.assertTrue(equal_matrices(DX, DX1))

        # test DZ using Z = z
        Z = z
        Y = softmax_colwise(Z)
        DY = substitute(diff(loss(y), y), y, Y)
        DZ = hadamard(Y, DY - row_repeat(diag(Y.T * DY).T, K))
        DZ1 = diff(loss(Y), z)
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

        # # feedforward
        X = x
        W = w
        Z = W * X + column_repeat(b, N)
        Y = log_softmax_colwise(Z)

        # # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DZ = DY - hadamard(softmax_colwise(Z), row_repeat(columns_sum(DY), K))
        DW = DZ * X.T
        Db = rows_sum(DZ)
        DX = W.T * DZ

        # test gradients
        DW1 = diff(loss(Y), w)
        Db1 = diff(loss(Y), b)
        DX1 = diff(loss(Y), x)
        self.assertTrue(equal_matrices(DW, DW1))
        self.assertTrue(equal_matrices(Db, Db1))
        self.assertTrue(equal_matrices(DX, DX1))

        # test DZ using Z = z
        Z = z
        Y = log_softmax_colwise(Z)
        DY = substitute(diff(loss(y), y), y, Y)
        DZ = DY - hadamard(softmax_colwise(Z), row_repeat(columns_sum(DY), K))
        DZ1 = diff(loss(Y), z)
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

        # feedforward
        X = x
        W = w
        Z = X * W.T + row_repeat(b, N)
        Y = softmax_rowwise(Z)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DZ = hadamard(Y, DY - column_repeat(diag(DY * Y.T), N))
        DW = DZ.T * X
        Db = columns_sum(DZ)
        DX = DZ * W

        # test gradients
        DW1 = diff(loss(Y), w)
        Db1 = diff(loss(Y), b)
        DX1 = diff(loss(Y), x)
        self.assertTrue(equal_matrices(DW, DW1))
        self.assertTrue(equal_matrices(Db, Db1))
        self.assertTrue(equal_matrices(DX, DX1))

        # test DZ using Z = z
        Z = z
        Y = softmax_rowwise(Z)
        DY = substitute(diff(loss(y), y), y, Y)
        DZ = hadamard(Y, DY - column_repeat(diag(DY * Y.T), N))
        DZ1 = diff(loss(Y), z)
        self.assertTrue(equal_matrices(DZ, DZ1))

    def test_log_softmax_layer_rowwise(self):
        D = 2
        K = 2
        N = 2
        loss = squared_error

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
        Y = log_softmax_rowwise(Z)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DZ = DY - hadamard(softmax_rowwise(Z), column_repeat(rows_sum(DY), N))
        DW = DZ.T * X
        Db = columns_sum(DZ)
        DX = DZ * W

        # test gradients
        DW1 = diff(loss(Y), w)
        Db1 = diff(loss(Y), b)
        DX1 = diff(loss(Y), x)
        # N.B. These tests take a long time...
        # self.assertTrue(equal_matrices(DW, DW1))
        # self.assertTrue(equal_matrices(Db, Db1))
        # self.assertTrue(equal_matrices(DX, DX1))

        # test DZ using Z = z
        Z = z
        Y = log_softmax_rowwise(Z)
        DY = substitute(diff(loss(y), y), y, Y)
        DZ = DY - hadamard(softmax_rowwise(Z), column_repeat(rows_sum(DY), N))
        DZ1 = diff(loss(Y), z)
        self.assertTrue(equal_matrices(DZ, DZ1))


if __name__ == '__main__':
    import unittest
    unittest.main()
