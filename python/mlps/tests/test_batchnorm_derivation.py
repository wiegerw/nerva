#!/usr/bin/env python3
# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase
from mlps.nerva_sympy.matrix_operations import *
from mlps.tests.utilities import matrix, pp, to_number

Matrix = sp.Matrix


class TestBatchNorm(TestCase):
    def test1(self):
        N = 2
        x = matrix('x', N, 1)
        r = matrix('r', N, 1)
        z = matrix('z', N, 1)

        X = x
        R = lambda X: (identity(N) - ones(N, N) / N) * X
        Sigma = lambda R: (R.T * R) / N
        Z = lambda R: to_number(power_minus_half(Sigma(R))) * R
        L = lambda Y: sp.Matrix([[elements_sum(Y)]])

        dz_dr = Z(r).jacobian(r)
        f_r = power_minus_half(Sigma(r))
        dz_dr2 = r * f_r.jacobian(r) + to_number(f_r) * sp.eye(N)
        self.assertEqual(dz_dr, dz_dr2)

        L_r =  L(Z(r))
        dL_dr = L_r.jacobian(r)
        pp('dL_dr', dL_dr)

        L_z = L(z)
        dL_dz = L_z.jacobian(z)
        pp('dL_dz', dL_dz)

        Dr = dL_dr
        Dz = substitute(dL_dz, (z, Z(r)))
        self.assertEqual(Dr, Dz * dz_dr)

        dsigma_dr = Sigma(r).jacobian(r)
        dsigma_dr1 = (2 * r.T) / N
        self.assertEqual(dsigma_dr, dsigma_dr1)

        df_dr = f_r.jacobian(r)
        power_minus_half_sigma_r = power_minus_half(Sigma(r))
        df_dr1 = - ((power_minus_half_sigma_r * power_minus_half_sigma_r * power_minus_half_sigma_r) / 2) * dsigma_dr
        self.assertEqual(df_dr, df_dr1)
        df_dr2 = - ((power_minus_half_sigma_r * power_minus_half_sigma_r * power_minus_half_sigma_r) / N) * r.T
        self.assertEqual(df_dr, df_dr2)

        dz_dr3 = r * df_dr2 + to_number(power_minus_half_sigma_r) * sp.eye(N)
        self.assertEqual(dz_dr, dz_dr3)

        z_r = Z(r)
        dz_dr4 = (to_number(power_minus_half_sigma_r) / N) * (-z_r * z_r.T + N * sp.eye(N))
        self.assertEqual(sp.simplify(dz_dr), sp.simplify(dz_dr4))


if __name__ == '__main__':
    import unittest
    unittest.main()
