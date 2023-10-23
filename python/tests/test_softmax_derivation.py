#!/usr/bin/env python3
# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase
from symbolic.sympy.matrix_operations import *
from symbolic.sympy.softmax_functions import *
from symbolic.sympy.loss_functions import *

import sympy as sp

from symbolic.utilities import equal_matrices

Matrix = sp.Matrix


def matrix(name: str, rows: int, columns: int) -> Matrix:
    return Matrix(sp.symarray(name, (rows, columns), real=True))


def pp(name: str, x: sp.Matrix):
    print(f'{name} ({x.shape[0]}x{x.shape[1]})')
    for row in x.tolist():
        print('[', end='')
        for i, elem in enumerate(row):
            print(f'{elem}', end='')
            if i < len(row) - 1:
                print(', ', end='')
        print(']')
    print()


def substitute(expr, substitutions: Union[Tuple[Matrix, Matrix], List[Tuple[Matrix, Matrix]]]):
    if isinstance(substitutions, tuple):
        substitutions = [substitutions]
    for (X, Y) in substitutions:
        assert X.shape == Y.shape
        m, n = X.shape
        sigma = ((X[i, j], Y[i, j]) for i in range(m) for j in range(n))
        expr = expr.subs(sigma)
    return expr


class TestSoftmaxDerivation(TestCase):
    def test1(self):
        K = 3
        z = matrix('z', 1, K)
        y = softmax_rowwise(z)
        self.assertEqual(softmax_rowwise(z).jacobian(z), Diag(y) - y.T * y)

    def test2(self):
        K = 2
        y = matrix('y', 1, K)
        z = matrix('z', 1, K)
        L1 = lambda Y: sp.Matrix([[elements_sum(Y)]])
        L2 = lambda Y: sp.Matrix([[elements_sum(hadamard(Y, Y))]])

        for L in [L1, L2]:
            y_z = softmax_rowwise(z)
            dsoftmax_z_dz = softmax_rowwise(z).jacobian(z)
            dL_dy = substitute(L(y).jacobian(y), (y, y_z))
            dL_dz = L(y_z).jacobian(z)
            self.assertTrue(equal_matrices(dL_dz, dL_dy * dsoftmax_z_dz))


if __name__ == '__main__':
    import unittest
    unittest.main()
