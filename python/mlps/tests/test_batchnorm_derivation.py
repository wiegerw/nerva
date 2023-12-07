#!/usr/bin/env python3
# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase

from mlps.nerva_sympy.matrix_operations import *
from mlps.tests.utilities import equal_matrices, matrix, to_matrix, to_number

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


class TestBatchNorm(TestCase):
    def test_derivation_Dx(self):
        N = 2
        x = matrix('x', N, 1)
        r = matrix('r', N, 1)
        z = matrix('z', N, 1)

        I = identity(N) - ones(N, N) / N
        R = lambda r: I * x
        Sigma = lambda r: (r.T * r) / N
        Z = lambda r: to_number(power_minus_half(Sigma(r))) * r
        L = lambda Y: to_matrix(squared_error_columns(Y))

        z_r = Z(r)
        dz_dr = z_r.jacobian(r)
        self.assertTrue(equal_matrices(dz_dr, to_number(power_minus_half(Sigma(r)) / N) * (N * identity(N) - z_r * z_r.T)))

        L_r = L(z_r)
        L_z = L(z)
        dL_dr = L_r.jacobian(r)
        dL_dz = L_z.jacobian(z)
        Dr = dL_dr.T
        Dz = substitute(dL_dz.T, (z, z_r))
        self.assertTrue(equal_matrices(Dr, to_number(power_minus_half(Sigma(r)) / N) * (N * identity(N) - z_r * z_r.T) * Dz))

        r_x = R(x)
        z_x = Z(r_x)
        L_x = L(z_x)
        Dx = L_x.jacobian(x).T
        Dr = substitute(Dr, (r, r_x))
        self.assertTrue(equal_matrices(Dx, I * Dr))

        sigma = Sigma(r_x)
        Dz = substitute(dL_dz.T, (z, z_x))
        z = z_x
        self.assertTrue(equal_matrices(Dx, to_number(power_minus_half(sigma) / N) * (N * I * Dz - z * z.T * Dz), simplify_arguments=True))

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
