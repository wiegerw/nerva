#!/usr/bin/env python3

# Copyright 2024 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

# see also https://docs.sympy.org/latest/modules/matrices/matrices.html

from unittest import TestCase

from mlps.nerva_sympy.matrix_operations import *
from mlps.tests.utilities import equal_matrices, matrix, to_matrix, to_number, pp, instantiate
import sympy as sp

Matrix = sp.Matrix


def squared_error_rows(X: Matrix):
    m, n = X.shape

    def f(x: Matrix) -> float:
        return sp.sqrt(sum(xj * xj for xj in x))

    return sum(f(X.row(i)) for i in range(m))


def squared_error_columns(X: Matrix):
    m, n = X.shape

    def f(x: Matrix) -> float:
        return sp.sqrt(sum(xj * xj for xj in x))

    return sum(f(X.col(j)) for j in range(n))


class TestLinearLayerDerivation(TestCase):
    def test_derivations(self):
        N = 3
        D = 4
        K = 2

        x = matrix('x', N, D)
        y = matrix('y', N, K)
        w = matrix('w', K, D)
        b = matrix('b', 1, K)
        X = x
        W = w
        B = row_repeat(b, N)

        # feedforward
        Y = X * W.T + row_repeat(b, N)

        L = lambda Y: to_matrix(squared_error_columns(Y))

        i = 1
        x_i = x.row(i)  # 1 x D
        y_i = y.row(i)  # 1 x K

        DY = substitute(gradient(L(y), y), (y, Y))
        DW = DY.T * X

        L_x = L(Y)
        L_y = L(y)

        dL_dx_i = L_x.jacobian(x_i)
        dL_dy_i = substitute(L_y.jacobian(y_i), [(y, Y)])
        dy_i_dx_i = Y.row(i).jacobian(x_i)

        # first derivation
        self.assertTrue(equal_matrices(dL_dx_i, dL_dy_i * dy_i_dx_i))
        self.assertTrue(equal_matrices(dL_dy_i * dy_i_dx_i, dL_dy_i * W))

        # second derivation
        dL_db = L_x.jacobian(b)
        sum_dL_dyi = substitute(sum([L_y.jacobian(y.row(i)) * Y.row(i).jacobian(b) for i in range(N)], sp.zeros(1, K)), [(y, Y)])
        self.assertTrue(equal_matrices(dL_db, sum_dL_dyi))
        for i in range(N):
            self.assertTrue(equal_matrices(Y.row(i).jacobian(b), sp.eye(K)))
        self.assertTrue(equal_matrices(sum_dL_dyi, columns_sum(DY)))

        # third derivation
        i = 1
        j = 2
        k = 1

        e_i = sp.Matrix([[1 if j == i else 0 for j in range(K)]]).T  # unit column vector with a 1 on the i-th position
        y_i = Y.row(i)  # 1 x K
        w_i = w.row(i)
        dyi_dwi = y_i.jacobian(w_i)
        self.assertTrue(equal_matrices(dyi_dwi, e_i * x_i))

        x_k = x.row(k)
        y_k = Y.row(k)
        self.assertTrue(equal_matrices(y_k, x_k * w.T + b))

        w_ij = to_matrix(w[i, j])
        dyk_dwij = y_k.jacobian(w_ij)
        self.assertTrue(equal_matrices(dyk_dwij, x[k, j] * e_i))


class TestBatchNormDerivation(TestCase):
    def test_derivation_Dx(self):
        N = 2
        x = matrix('x', N, 1)
        r = matrix('r', N, 1)
        z = matrix('z', N, 1)

        I = identity(N) - ones(N, N) / N
        R = lambda r: I * x
        Sigma = lambda r: (r.T * r) / N
        Z = lambda r: to_number(inv_sqrt(Sigma(r))) * r
        L = lambda Y: to_matrix(squared_error_columns(Y))

        z_r = Z(r)
        dz_dr = z_r.jacobian(r)
        self.assertTrue(equal_matrices(dz_dr, to_number(inv_sqrt(Sigma(r)) / N) * (N * identity(N) - z_r * z_r.T)))

        L_r = L(z_r)
        L_z = L(z)
        dL_dr = L_r.jacobian(r)
        dL_dz = L_z.jacobian(z)
        Dr = dL_dr.T
        Dz = substitute(dL_dz.T, (z, z_r))
        self.assertTrue(equal_matrices(Dr, to_number(inv_sqrt(Sigma(r)) / N) * (N * identity(N) - z_r * z_r.T) * Dz))

        r_x = R(x)
        z_x = Z(r_x)
        L_x = L(z_x)
        Dx = L_x.jacobian(x).T
        Dr = substitute(Dr, (r, r_x))
        self.assertTrue(equal_matrices(Dx, I * Dr))

        sigma = Sigma(r_x)
        Dz = substitute(dL_dz.T, (z, z_x))
        z = z_x
        self.assertTrue(equal_matrices(Dx, to_number(inv_sqrt(sigma) / N) * (N * I * Dz - z * z.T * Dz), simplify_arguments=True))

    def test_derivation_DW(self):
        N = 2
        K = 2
        D = 3
        x = matrix('x', N, D)
        y = matrix('y', N, K)
        w = matrix('w', K, D)
        b = matrix('b', 1, K)
        L = lambda Y: to_matrix(squared_error_rows(Y))
        I = identity(N)
        i = 1
        j = 1
        w_j = w.row(j)
        x_i = x.row(i)
        e_j = I.col(j)

        Y = x * w.T + row_repeat(b, N)
        y_i = Y.row(i)

        Dw_j = L(y_i).jacobian(w_j)
        Dy_i = substitute(L(y).jacobian(y.row(i)), (y.row(i), y_i))
        dyi_dwj = y_i.jacobian(w.row(j))

        self.assertTrue(equal_matrices(dyi_dwj, e_j * x_i))
        self.assertTrue(equal_matrices(Dw_j, Dy_i * e_j * x_i))

        DW = Matrix([L(y_i).jacobian(w.row(j)) for j in range(K)])
        self.assertTrue(equal_matrices(DW, Dy_i.T * x_i))


if __name__ == '__main__':
    import unittest
    unittest.main()
