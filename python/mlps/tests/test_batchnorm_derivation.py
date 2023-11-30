#!/usr/bin/env python3
# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase

from mlps.nerva_sympy.matrix_operations import *
from mlps.tests.utilities import matrix, pp, to_number, squared_error, to_matrix, equal_matrices

Matrix = sp.Matrix


class TestBatchNorm(TestCase):
    def test_derivation(self):
        N = 2
        x = matrix('x', N, 1)
        r = matrix('r', N, 1)
        z = matrix('z', N, 1)

        I = identity(N) - ones(N, N) / N
        R = lambda r: I * x
        Sigma = lambda r: (r.T * r) / N
        Z = lambda r: to_number(power_minus_half(Sigma(r))) * r
        L = lambda Y: to_matrix(squared_error(Y))

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


if __name__ == '__main__':
    import unittest
    unittest.main()
