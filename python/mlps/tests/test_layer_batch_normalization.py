#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

# see also https://docs.sympy.org/latest/modules/matrices/matrices.html

from unittest import TestCase

from mlps.nerva_sympy.matrix_operations import *
from mlps.tests.utilities import equal_matrices, matrix, squared_error


class TestBatchNormalizationLayers(TestCase):

    def test_simple_batch_normalization_layer_colwise(self):
        D = 3
        N = 2
        K = D                # K and D are always equal in batch normalization
        loss = elements_sum  # squared_error seems too complicated

        # variables
        x = matrix('x', D, N)
        y = matrix('y', K, N)
        X = x

        # feedforward
        R = X - column_repeat(rows_mean(X), N)
        Sigma = diag(R * R.T) / N
        inv_sqrt_Sigma = inv_sqrt(Sigma)
        Y = hadamard(column_repeat(inv_sqrt_Sigma, N), R)

        # symbolic differentiation
        DY = substitute(gradient(loss(y), y), (y, Y))

        # backpropagation
        DX = hadamard(column_repeat(inv_sqrt_Sigma / N, N), hadamard(Y, column_repeat(-diag(DY * Y.T), N)) + DY * (N * identity(N) - ones(N, N)))

        # test gradients
        DX1 = gradient(loss(Y), x)

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
        X = x

        # feedforward
        Y = hadamard(column_repeat(gamma, N), X) + column_repeat(beta, N)

        # symbolic differentiation
        DY = substitute(gradient(loss(y), y), (y, Y))

        # backpropagation
        DX = hadamard(column_repeat(gamma, N), DY)
        Dbeta = rows_sum(DY)
        Dgamma = rows_sum(hadamard(X, DY))

        # test gradients
        DX1 = gradient(loss(Y), x)
        Dbeta1 = gradient(loss(Y), beta)
        Dgamma1 = gradient(loss(Y), gamma)

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
        X = x

        # feedforward
        R = X - column_repeat(rows_mean(X), N)
        Sigma = diag(R * R.T) / N
        inv_sqrt_Sigma = inv_sqrt(Sigma)
        Z = hadamard(column_repeat(inv_sqrt_Sigma, N), R)
        Y = hadamard(column_repeat(gamma, N), Z) + column_repeat(beta, N)

        # symbolic differentiation
        DY = substitute(gradient(loss(y), y), (y, Y))

        # backpropagation
        DZ = hadamard(column_repeat(gamma, N), DY)
        Dbeta = rows_sum(DY)
        Dgamma = rows_sum(hadamard(DY, Z))
        DX = hadamard(column_repeat(inv_sqrt_Sigma / N, N), hadamard(Z, column_repeat(-diag(DZ * Z.T), N)) + DZ * (N * identity(N) - ones(N, N)))

        # test gradients
        DX1 = gradient(loss(Y), x)
        Dbeta1 = gradient(loss(Y), beta)
        Dgamma1 = gradient(loss(Y), gamma)
        Y_z = hadamard(column_repeat(gamma, N), z) + column_repeat(beta, N)
        DZ1 = substitute(gradient(loss(Y_z), z), (z, Z))

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
        X = x

        # feedforward
        R = X - row_repeat(columns_mean(X), N)
        Sigma = diag(R.T * R).T / N
        inv_sqrt_Sigma = inv_sqrt(Sigma)
        Y = hadamard(row_repeat(inv_sqrt_Sigma, N), R)

        # symbolic differentiation
        DY = substitute(gradient(loss(y), y), (y, Y))

        # backpropagation
        DX = hadamard(row_repeat(inv_sqrt_Sigma / N, N), (N * identity(N) - ones(N, N)) * DY - hadamard(Y, row_repeat(diag(Y.T * DY).T, N)))

        # test gradients
        DX1 = gradient(loss(Y), x)

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
        X = x

        # feedforward
        Y = hadamard(row_repeat(gamma, N), X) + row_repeat(beta, N)

        # symbolic differentiation
        DY = substitute(gradient(loss(y), y), (y, Y))

        # backpropagation
        DX = hadamard(row_repeat(gamma, N), DY)
        Dbeta = columns_sum(DY)
        Dgamma = columns_sum(hadamard(X, DY))

        # test gradients
        DX1 = gradient(loss(Y), x)
        Dbeta1 = gradient(loss(Y), beta)
        Dgamma1 = gradient(loss(Y), gamma)

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
        X = x

        # feedforward
        R = X - row_repeat(columns_mean(X), N)
        Sigma = diag(R.T * R).T / N
        inv_sqrt_Sigma = inv_sqrt(Sigma)
        Z = hadamard(row_repeat(inv_sqrt_Sigma, N), R)
        Y = hadamard(row_repeat(gamma, N), Z) + row_repeat(beta, N)

        # symbolic differentiation
        DY = substitute(gradient(loss(y), y), (y, Y))

        # backpropagation
        DZ = hadamard(row_repeat(gamma, N), DY)
        Dbeta = columns_sum(DY)
        Dgamma = columns_sum(hadamard(DY, Z))
        DX = hadamard(row_repeat(inv_sqrt_Sigma / N, N), (N * identity(N) - ones(N, N)) * DZ - hadamard(Z, row_repeat(diag(Z.T * DZ).T, N)))

        # test gradients
        DX1 = gradient(loss(Y), x)
        Dbeta1 = gradient(loss(Y), beta)
        Dgamma1 = gradient(loss(Y), gamma)
        Y_z = hadamard(row_repeat(gamma, N), z) + row_repeat(beta, N)
        DZ1 = substitute(gradient(loss(Y_z), z), (z, Z))

        self.assertTrue(equal_matrices(DX, DX1))
        self.assertTrue(equal_matrices(Dbeta, Dbeta1))
        self.assertTrue(equal_matrices(Dgamma, Dgamma1))
        self.assertTrue(equal_matrices(DZ, DZ1))

    def test_yeh_batch_normalization_layer_rowwise(self):
        # see https://chrisyeh96.github.io/2017/08/28/deriving-batchnorm-backprop.html

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
        X = x

        # feedforward
        R = X - row_repeat(columns_mean(X), N)
        Sigma = diag(R.T * R).T / N
        inv_sqrt_Sigma = inv_sqrt(Sigma)
        Z = hadamard(row_repeat(inv_sqrt_Sigma, N), R)
        Y = hadamard(row_repeat(gamma, N), Z) + row_repeat(beta, N)

        # symbolic differentiation
        DY = substitute(gradient(loss(y), y), (y, Y))

        # backpropagation
        DZ = hadamard(row_repeat(gamma, N), DY)  # this equation is not explicitly given in [Yeh 2017]
        Dbeta = columns_sum(DY)                  # this equation is the same as in [Yeh 2017]
        Dgamma = columns_sum(hadamard(DY, Z))    # I can't parse the equation in [Yeh 2017], but this is probably it
        DX = (1 / N) * (-hadamard(row_repeat(Dgamma, N), Z) + N * DY - row_repeat(Dbeta, N)) * row_repeat(hadamard(gamma, Sigma), D) # I can't parse the equation in [Yeh 2017], but this is probably it

        # test gradients
        DX1 = gradient(loss(Y), x)
        Dbeta1 = gradient(loss(Y), beta)
        Dgamma1 = gradient(loss(Y), gamma)
        Y_z = hadamard(row_repeat(gamma, N), z) + row_repeat(beta, N)
        DZ1 = substitute(gradient(loss(Y_z), z), (z, Z))

        self.assertTrue(equal_matrices(DX, DX1))
        self.assertTrue(equal_matrices(Dbeta, Dbeta1))
        self.assertTrue(equal_matrices(Dgamma, Dgamma1))
        self.assertTrue(equal_matrices(DZ, DZ1))


if __name__ == '__main__':
    import unittest
    unittest.main()
