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
        loss = sum_elements  # squared_error seems too complicated

        # variables
        x = matrix('x', D, N)
        y = matrix('y', K, N)

        # feedforward
        X = x
        R = X * (identity(N) - ones(N, N) / N)
        Sigma = diag(R * R.T) / N
        power_minus_half_Sigma = power_minus_half(Sigma)
        Y = hadamard(repeat_column(power_minus_half_Sigma, N), R)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DX = hadamard(repeat_column(power_minus_half_Sigma / N, N), hadamard(Y, repeat_column(-diag(DY * Y.T), N)) + DY * (N * identity(N) - ones(N, N)))

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
        Y = hadamard(repeat_column(gamma, N), X) + repeat_column(beta, N)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DX = hadamard(repeat_column(gamma, N), DY)
        Dbeta = sum_rows(DY)
        Dgamma = sum_rows(hadamard(X, DY))

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
        loss = sum_elements  # squared_error seems too complicated

        # variables
        x = matrix('x', D, N)
        y = matrix('y', K, N)
        z = matrix('z', K, N)
        beta = matrix('beta', K, 1)
        gamma = matrix('gamma', K, 1)

        # feedforward
        X = x
        R = X * (identity(N) - ones(N, N) / N)
        Sigma = diag(R * R.T) / N
        power_minus_half_Sigma = power_minus_half(Sigma)
        Z = hadamard(repeat_column(power_minus_half_Sigma, N), R)
        Y = hadamard(repeat_column(gamma, N), Z) + repeat_column(beta, N)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DZ = hadamard(repeat_column(gamma, N), DY)
        Dbeta = sum_rows(DY)
        Dgamma = sum_rows(hadamard(DY, Z))
        DX = hadamard(repeat_column(power_minus_half_Sigma / N, N), hadamard(Z, repeat_column(-diag(DZ * Z.T), N)) + DZ * (N * identity(N) - ones(N, N)))

        # test gradients
        DX1 = diff(loss(Y), x)
        Dbeta1 = diff(loss(Y), beta)
        Dgamma1 = diff(loss(Y), gamma)
        Y_z = hadamard(repeat_column(gamma, N), z) + repeat_column(beta, N)
        DZ1 = substitute(diff(loss(Y_z), z), z, Z)

        self.assertTrue(equal_matrices(DX, DX1))
        self.assertTrue(equal_matrices(Dbeta, Dbeta1))
        self.assertTrue(equal_matrices(Dgamma, Dgamma1))
        self.assertTrue(equal_matrices(DZ, DZ1))

    def test_simple_batch_normalization_layer_rowwise(self):
        D = 3
        N = 2
        K = D                # K and D are always equal in batch normalization
        loss = sum_elements  # squared_error seems too complicated

        # variables
        x = matrix('x', N, D)
        y = matrix('y', N, K)

        # feedforward
        X = x
        R = (identity(N) - ones(N, N) / N) * X
        Sigma = diag(R.T * R).T / N
        power_minus_half_Sigma = power_minus_half(Sigma)
        Y = hadamard(repeat_row(power_minus_half_Sigma, N), R)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DX = hadamard(repeat_row(power_minus_half_Sigma / N, N),
                      (N * identity(N) - ones(N, N)) * DY - hadamard(Y, repeat_row(diag(Y.T * DY).T, N)))

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
        Y = hadamard(repeat_row(gamma, N), X) + repeat_row(beta, N)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DX = hadamard(repeat_row(gamma, N), DY)
        Dbeta = sum_columns(DY)
        Dgamma = sum_columns(hadamard(X, DY))

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
        loss = sum_elements  # squared_error seems too complicated

        # variables
        x = matrix('x', N, D)
        y = matrix('y', N, K)
        z = matrix('z', N, K)
        beta = matrix('beta', 1, K)
        gamma = matrix('gamma', 1, K)

        # feedforward
        X = x
        R = (identity(N) - ones(N, N) / N) * X
        Sigma = diag(R.T * R).T / N
        power_minus_half_Sigma = power_minus_half(Sigma)
        Z = hadamard(repeat_row(power_minus_half_Sigma, N), R)
        Y = hadamard(repeat_row(gamma, N), Z) + repeat_row(beta, N)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DZ = hadamard(repeat_row(gamma, N), DY)
        Dbeta = sum_columns(DY)
        Dgamma = sum_columns(hadamard(Z, DY))
        DX = hadamard(repeat_row(power_minus_half_Sigma / N, N),
                      (N * identity(N) - ones(N, N)) * DZ - hadamard(Z, repeat_row(diag(Z.T * DZ).T, N)))

        # test gradients
        DX1 = diff(loss(Y), x)
        Dbeta1 = diff(loss(Y), beta)
        Dgamma1 = diff(loss(Y), gamma)
        Y_z = hadamard(repeat_row(gamma, N), z) + repeat_row(beta, N)
        DZ1 = substitute(diff(loss(Y_z), z), z, Z)

        self.assertTrue(equal_matrices(DX, DX1))
        self.assertTrue(equal_matrices(Dbeta, Dbeta1))
        self.assertTrue(equal_matrices(Dgamma, Dgamma1))
        self.assertTrue(equal_matrices(DZ, DZ1))


if __name__ == '__main__':
    import unittest
    unittest.main()
