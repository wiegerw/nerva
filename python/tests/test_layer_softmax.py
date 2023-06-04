#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase
from symbolic.loss_functions import *
from symbolic.softmax import *
from symbolic.utilities import *


class TestSoftmaxLayers(TestCase):

    def test_softmax_layer_colwise(self):
        K = 3
        N = 2
        loss = squared_error

        # variables
        y = matrix('y', K, N)
        z = matrix('z', K, N)

        # feedforward
        Z = z
        Y = softmax_colwise(Z)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DZ = hadamard(Y, DY - repeat_row(diag(Y.T * DY).T, K))

        # symbolic differentiation
        DZ1 = diff(loss(Y), z)

        self.assertTrue(equal_matrices(DZ, DZ1))

    def test_log_softmax_layer_colwise(self):
        K = 3
        N = 2
        loss = squared_error

        # variables
        y = matrix('y', K, N)
        z = matrix('z', K, N)

        # feedforward
        Z = z
        Y = log_softmax_colwise(Z)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DZ = DY - hadamard(softmax_colwise(Z), repeat_row(sum_columns(DY), K))

        # symbolic differentiation
        DZ1 = diff(loss(Y), z)

        self.assertTrue(equal_matrices(DZ, DZ1))

    def test_softmax_layer_rowwise(self):
        K = 2
        N = 3
        loss = squared_error

        # variables
        y = matrix('y', K, N)
        z = matrix('z', K, N)

        # feedforward
        Z = z
        Y = softmax_rowwise(Z)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DZ = hadamard(Y, DY - repeat_column(diag(DY * Y.T), N))

        # symbolic differentiation
        DZ1 = diff(loss(Y), z)

        self.assertTrue(equal_matrices(DZ, DZ1))

    def test_log_softmax_layer_rowwise(self):
        K = 2
        N = 3
        loss = squared_error

        # variables
        y = matrix('y', K, N)
        z = matrix('z', K, N)

        # feedforward
        Z = z
        Y = log_softmax_rowwise(Z)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DZ = DY - hadamard(softmax_rowwise(Z), repeat_column(sum_rows(DY), N))

        # symbolic differentiation
        DZ1 = diff(loss(Y), z)

        self.assertTrue(equal_matrices(DZ, DZ1))


if __name__ == '__main__':
    import unittest
    unittest.main()
