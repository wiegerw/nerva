#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

# see also https://docs.sympy.org/latest/modules/matrices/matrices.html

from unittest import TestCase
from symbolic.loss_functions import *
from symbolic.matrix_operations import *
from symbolic.utilities import *


class TestBatchNormalizationLayers(TestCase):

    def test_simple_batch_normalization_layer_colwise(self):
        D = 3
        N = 2
        K = D                # K and D are always equal in batch normalization
        loss = elements_sum  # squared_error seems too complicated

        # variables
        x = matrix('x', D, N)
        y = matrix('y', K, N)

        # feedforward
        X = x
        R = X - column_repeat(rows_mean(X), N)
        Sigma = diag(R * R.T) / N
        power_minus_half_Sigma = power_minus_half(Sigma)
        Y = hadamard(column_repeat(power_minus_half_Sigma, N), R)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DX = hadamard(column_repeat(power_minus_half_Sigma / N, N), hadamard(Y, column_repeat(-diag(DY * Y.T), N)) + DY * (N * identity(N) - ones(N, N)))

        # test gradients
        DX1 = diff(loss(Y), x)

        self.assertTrue(equal_matrices(DX, DX1))

    def test_affine_layer_colwise(self):
        D = 3
        N = 2
        K = D
        loss = squared_error

        # variables
        x = matrix('x', D, N)
        y = matrix('y', K, N)
        beta = matrix('beta', K, 1)
        gamma = matrix('gamma', K, 1)

        # feedforward
        X = x
        Y = hadamard(column_repeat(gamma, N), X) + column_repeat(beta, N)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DX = hadamard(column_repeat(gamma, N), DY)
        Dbeta = rows_sum(DY)
        Dgamma = rows_sum(hadamard(X, DY))

        # test gradients
        DX1 = diff(loss(Y), x)
        Dbeta1 = diff(loss(Y), beta)
        Dgamma1 = diff(loss(Y), gamma)

        self.assertTrue(equal_matrices(DX, DX1))
        self.assertTrue(equal_matrices(Dbeta, Dbeta1))
        self.assertTrue(equal_matrices(Dgamma, Dgamma1))

    def test_batch_normalization_layer_colwise(self):
        D = 3
        N = 2
        K = D                # K and D are always equal in batch normalization
        loss = elements_sum  # squared_error seems too complicated

        # variables
        x = matrix('x', D, N)
        y = matrix('y', K, N)
        z = matrix('z', K, N)
        beta = matrix('beta', K, 1)
        gamma = matrix('gamma', K, 1)

        # feedforward
        X = x
        R = X - column_repeat(rows_mean(X), N)
        Sigma = diag(R * R.T) / N
        power_minus_half_Sigma = power_minus_half(Sigma)
        Z = hadamard(column_repeat(power_minus_half_Sigma, N), R)
        Y = hadamard(column_repeat(gamma, N), Z) + column_repeat(beta, N)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DZ = hadamard(column_repeat(gamma, N), DY)
        Dbeta = rows_sum(DY)
        Dgamma = rows_sum(hadamard(DY, Z))
        DX = hadamard(column_repeat(power_minus_half_Sigma / N, N), hadamard(Z, column_repeat(-diag(DZ * Z.T), N)) + DZ * (N * identity(N) - ones(N, N)))

        # test gradients
        DX1 = diff(loss(Y), x)
        Dbeta1 = diff(loss(Y), beta)
        Dgamma1 = diff(loss(Y), gamma)
        Y_z = hadamard(column_repeat(gamma, N), z) + column_repeat(beta, N)
        DZ1 = substitute(diff(loss(Y_z), z), z, Z)

        self.assertTrue(equal_matrices(DX, DX1))
        self.assertTrue(equal_matrices(Dbeta, Dbeta1))
        self.assertTrue(equal_matrices(Dgamma, Dgamma1))
        self.assertTrue(equal_matrices(DZ, DZ1))

    def test_simple_batch_normalization_layer_rowwise(self):
        D = 3
        N = 2
        K = D                # K and D are always equal in batch normalization
        loss = elements_sum  # squared_error seems too complicated

        # variables
        x = matrix('x', N, D)
        y = matrix('y', N, K)

        # feedforward
        X = x
        R = X - row_repeat(columns_mean(X), N)
        Sigma = diag(R.T * R).T / N
        power_minus_half_Sigma = power_minus_half(Sigma)
        Y = hadamard(row_repeat(power_minus_half_Sigma, N), R)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DX = hadamard(row_repeat(power_minus_half_Sigma / N, N), (N * identity(N) - ones(N, N)) * DY - hadamard(Y, row_repeat(diag(Y.T * DY).T, N)))

        # test gradients
        DX1 = diff(loss(Y), x)

        self.assertTrue(equal_matrices(DX, DX1))

    def test_affine_layer_rowwise(self):
        D = 3
        N = 2
        K = D
        loss = squared_error

        # variables
        x = matrix('x', N, D)
        y = matrix('y', N, K)
        beta = matrix('beta', 1, K)
        gamma = matrix('gamma', 1, K)

        # feedforward
        X = x
        Y = hadamard(row_repeat(gamma, N), X) + row_repeat(beta, N)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DX = hadamard(row_repeat(gamma, N), DY)
        Dbeta = columns_sum(DY)
        Dgamma = columns_sum(hadamard(X, DY))

        # test gradients
        DX1 = diff(loss(Y), x)
        Dbeta1 = diff(loss(Y), beta)
        Dgamma1 = diff(loss(Y), gamma)

        self.assertTrue(equal_matrices(DX, DX1))
        self.assertTrue(equal_matrices(Dbeta, Dbeta1))
        self.assertTrue(equal_matrices(Dgamma, Dgamma1))

    def test_batch_normalization_layer_rowwise(self):
        D = 3
        N = 2
        K = D                # K and D are always equal in batch normalization
        loss = elements_sum  # squared_error seems too complicated

        # variables
        x = matrix('x', N, D)
        y = matrix('y', N, K)
        z = matrix('z', N, K)
        beta = matrix('beta', 1, K)
        gamma = matrix('gamma', 1, K)

        # feedforward
        X = x
        R = X - row_repeat(columns_mean(X), N)
        Sigma = diag(R.T * R).T / N
        power_minus_half_Sigma = power_minus_half(Sigma)
        Z = hadamard(row_repeat(power_minus_half_Sigma, N), R)
        Y = hadamard(row_repeat(gamma, N), Z) + row_repeat(beta, N)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DZ = hadamard(row_repeat(gamma, N), DY)
        Dbeta = columns_sum(DY)
        Dgamma = columns_sum(hadamard(Z, DY))
        DX = hadamard(row_repeat(power_minus_half_Sigma / N, N), (N * identity(N) - ones(N, N)) * DZ - hadamard(Z, row_repeat(diag(Z.T * DZ).T, N)))

        # test gradients
        DX1 = diff(loss(Y), x)
        Dbeta1 = diff(loss(Y), beta)
        Dgamma1 = diff(loss(Y), gamma)
        Y_z = hadamard(row_repeat(gamma, N), z) + row_repeat(beta, N)
        DZ1 = substitute(diff(loss(Y_z), z), z, Z)

        self.assertTrue(equal_matrices(DX, DX1))
        self.assertTrue(equal_matrices(Dbeta, Dbeta1))
        self.assertTrue(equal_matrices(Dgamma, Dgamma1))
        self.assertTrue(equal_matrices(DZ, DZ1))


if __name__ == '__main__':
    import unittest
    unittest.main()
