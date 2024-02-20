#!/usr/bin/env python3
# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase

from mlps.nerva_sympy.matrix_operations import *
from mlps.nerva_sympy.softmax_functions import *
from mlps.tests.utilities import equal_matrices, matrix, to_matrix, to_number, pp

Matrix = sp.Matrix


class TestSoftmaxDerivation(TestCase):
    # section 4
    def test_dsoftmax_dz_derivation(self):
        K = 3

        def softmax(z: Matrix) -> Matrix:
            return reciprocal(rows_sum(exp(z))) * exp(z)

        z = matrix('z', 1, K)
        y = softmax(z)

        dsoftmax_dz = softmax(z).jacobian(z)
        lhs = exp(z)
        rhs = reciprocal(rows_sum(exp(z)))
        dlhs_dz = lhs.jacobian(z)
        drhs_dz = rhs.jacobian(z)
        self.assertTrue(equal_matrices(dsoftmax_dz, dlhs_dz * to_number(rhs) + lhs.T * drhs_dz))
        self.assertTrue(equal_matrices(dlhs_dz * to_number(rhs), Diag(exp(z)) * to_number(rhs)))
        R = to_number(rows_sum(exp(z)))
        self.assertTrue(equal_matrices(drhs_dz, - exp(z) / (R * R)))
        self.assertTrue(equal_matrices(Diag(exp(z)) * to_number(rhs), Diag(y)))
        self.assertTrue(equal_matrices(exp(z).T * (exp(z) / (R * R)), y.T * y))
        self.assertTrue(equal_matrices(dsoftmax_dz, Diag(y) - y.T * y))

    # appendix C.2
    def test_softmax_derivation(self):
        D = 3

        def softmax(x: Matrix) -> Matrix:
            return reciprocal(rows_sum(exp(x))) * exp(x)

        x = matrix('x', 1, D)
        y = softmax(x)

        E = exp(x)
        Q = rows_sum(exp(x))
        dE_dx = E.jacobian(x)
        dQ_dx = Q.jacobian(x)
        self.assertTrue(equal_matrices(dQ_dx, columns_sum(dE_dx)))
        self.assertTrue(equal_matrices(columns_sum(dE_dx), columns_sum(Diag(exp(x)))))
        self.assertTrue(equal_matrices(columns_sum(Diag(exp(x))), exp(x)))

        dsoftmax_dx = softmax(x).jacobian(x)
        lhs = exp(x)
        rhs = reciprocal(rows_sum(exp(x)))
        dlhs_dx = lhs.jacobian(x)
        drhs_dx = rhs.jacobian(x)
        self.assertTrue(equal_matrices(dsoftmax_dx, dlhs_dx * to_number(rhs) + lhs.T * drhs_dx))
        self.assertTrue(equal_matrices(dlhs_dx * to_number(rhs), Diag(exp(x)) * to_number(rhs)))
        R = to_number(rows_sum(exp(x)))
        self.assertTrue(equal_matrices(drhs_dx, - exp(x) / (R * R)))
        self.assertTrue(equal_matrices(Diag(exp(x)) * to_number(rhs), Diag(y)))
        self.assertTrue(equal_matrices(exp(x).T * (exp(x) / (R * R)), y.T * y))
        self.assertTrue(equal_matrices(dsoftmax_dx, Diag(y) - y.T * y))

    # appendix C.2
    def test_log_softmax_derivation(self):
        D = 3

        def softmax(x: Matrix) -> Matrix:
            return reciprocal(rows_sum(exp(x))) * exp(x)

        def log_softmax(x: Matrix) -> Matrix:
            return log(softmax(x))

        x = matrix('x', 1, D)
        y = softmax(x)
        z = matrix('z', 1, D)

        dsoftmax_dx = softmax(x).jacobian(x)
        dlog_softmax_dx = log_softmax(x).jacobian(x)
        dlog_z_dz = substitute(log(z).jacobian(z), (z, y))

        self.assertTrue(equal_matrices(dlog_softmax_dx, dlog_z_dz * dsoftmax_dx))
        self.assertTrue(equal_matrices(dlog_z_dz, Diag(reciprocal(y))))
        self.assertTrue(equal_matrices(dsoftmax_dx, Diag(y) - y.T * y))
        self.assertTrue(equal_matrices(dlog_softmax_dx, Diag(reciprocal(y)) * (Diag(y) - y.T * y)))
        self.assertTrue(equal_matrices(dlog_softmax_dx, identity(D) - Diag(reciprocal(y)) * y.T * y))
        self.assertTrue(equal_matrices(dlog_softmax_dx, identity(D) - row_repeat(y, D)))

    def test1(self):
        K = 3
        z = matrix('z', 1, K)
        y = softmax_rowwise(z)
        self.assertTrue(equal_matrices(softmax_rowwise(z).jacobian(z), Diag(y) - y.T * y))

    def test2(self):
        K = 3
        y = matrix('y', 1, K)
        z = matrix('z', 1, K)
        L1 = lambda Y: to_matrix(elements_sum(Y))
        L2 = lambda Y: to_matrix(elements_sum(hadamard(Y, Y)))

        for L in [L1, L2]:
            y_z = softmax_rowwise(z)
            dsoftmax_z_dz = softmax_rowwise(z).jacobian(z)
            dL_dy = substitute(L(y).jacobian(y), (y, y_z))
            dL_dz = L(y_z).jacobian(z)
            self.assertTrue(equal_matrices(dL_dz, dL_dy * dsoftmax_z_dz))

    def test3(self):
        K = 3
        y = matrix('y', 1, K)
        z = matrix('z', 1, K)
        L1 = lambda Y: to_matrix(elements_sum(Y))
        L2 = lambda Y: to_matrix(elements_sum(hadamard(Y, Y)))

        for L in [L1, L2]:
            y_z = softmax_rowwise(z)
            dsoftmax_z_dz = y_z.jacobian(z)
            dL_dy = substitute(L(y).jacobian(y), (y, y_z))
            dL_dz = L(y_z).jacobian(z)
            Dy = dL_dy
            Dz = dL_dz
            self.assertTrue(equal_matrices(Dz, Dy * dsoftmax_z_dz))
            self.assertTrue(equal_matrices(dsoftmax_z_dz, Diag(y_z) - y_z.T * y_z))
            self.assertTrue(equal_matrices(Dz, Dy * (Diag(y_z) - y_z.T * y_z)))
            self.assertTrue(equal_matrices(Dz, hadamard(y_z, Dy) - Dy * y_z.T * y_z))

    def test4(self):
        D = 2
        x = matrix('x', 1, D)
        y = rows_sum(exp(x))
        dy_dx = jacobian(y, x)
        dy_dx_expected = exp(x)
        self.assertEqual(dy_dx_expected, dy_dx)

    def test5(self):
        D = 2
        x = matrix('x', 1, D)
        y = log(rows_sum(exp(x)))
        dy_dx = jacobian(y, x)
        dy_dx_expected = reciprocal(rows_sum(exp(x))) * exp(x)
        self.assertEqual(dy_dx_expected, dy_dx)

    def test6(self):
        D = 2
        x = matrix('x', 1, D)
        y = log(reciprocal(rows_sum(exp(x))) * exp(x))
        y1 = x - log(rows_sum(exp(x)))  * ones(1, D)
        y = sp.simplify(y)
        y1 = sp.simplify(y1)
        self.assertEqual(sp.simplify(y), sp.simplify(y1))

    def test7(self):
        D = 2
        x = matrix('x', 1, D)
        y = log(reciprocal(rows_sum(exp(x))) * exp(x))
        y = sp.simplify(y)
        dy_dx = jacobian(y, x)
        dy_dx_expected = identity(D) - row_repeat(reciprocal(rows_sum(exp(x))) * exp(x), D)
        dy_dx = sp.simplify(dy_dx)
        dy_dx_expected = sp.simplify(dy_dx_expected)
        self.assertEqual(dy_dx_expected, dy_dx)

    def test_example(self):
        from sympy import symarray
        softmax = softmax_rowwise
        K = 3
        z = Matrix(symarray('z', (1, K), real=True))  # create a symbolic 1xK vector
        y = softmax(z)
        e = exp(z)
        f = rows_sum(e)[0, 0]  # rows_sum returns a Matrix, so we extract the value
        self.assertTrue(equal_matrices(softmax(z).jacobian(z), Diag(e) / f - (e.T * e) / (f * f)))
        self.assertTrue(equal_matrices(softmax(z).jacobian(z), Diag(y) - y.T * y))


if __name__ == '__main__':
    import unittest
    unittest.main()
