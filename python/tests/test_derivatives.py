#!/usr/bin/env python3
# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from typing import Union, Tuple, List
from unittest import TestCase

import sympy as sp

Matrix = sp.Matrix

def matrix(name: str, rows: int, columns: int) -> Matrix:
    return Matrix(sp.symarray(name, (rows, columns), real=True))


def is_column_vector(x: Matrix) -> bool:
    m, n = x.shape
    return n == 1


def is_row_vector(x: Matrix) -> bool:
    m, n = x.shape
    return m == 1


def is_square(X: Matrix) -> bool:
    m, n = X.shape
    return m == n


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


def diag(X: Matrix) -> Matrix:
    assert is_square(X)
    m, n = X.shape
    return Matrix([[X[i, i] for i in range(m)]]).T


def Diag(x: Matrix) -> Matrix:
    assert is_column_vector(x) or is_row_vector(x)
    return sp.diag(*x)


def substitute(expr, substitutions: Union[Tuple[Matrix, Matrix], List[Tuple[Matrix, Matrix]]]):
    if isinstance(substitutions, tuple):
        substitutions = [substitutions]
    for (X, Y) in substitutions:
        assert X.shape == Y.shape
        m, n = X.shape
        sigma = ((X[i, j], Y[i, j]) for i in range(m) for j in range(n))
        expr = expr.subs(sigma)
    return expr


class TestLemmas(TestCase):
    def test_chain_rule_colwise(self):
        x = matrix('x', 3, 1)
        y = matrix('y', 3, 1)
        c = Matrix([[2, 3, 4]])
        f_x = Diag(c) * x + Matrix([[x[1, 0]], [x[1, 0]], [x[0, 0]]])
        g_y = y.T * y
        g_fx = substitute(g_y, (y, f_x))

        pp('x', x)
        pp('f(x)', f_x)
        pp('g(f(x))', g_fx)
        pp('g(y)', g_y)
        pp('f(x).jacobian(x)', f_x.jacobian(x))
        pp('g(f(x)).jacobian(x)', g_fx.jacobian(x))
        pp('g(y).jacobian(y)', g_y.jacobian(y))

        df_dx = f_x.jacobian(x)
        dg_dy = substitute(g_y.jacobian(y), (y, f_x))
        dg_dx = g_fx.jacobian(x)

        pp('dg_dx', dg_dx)
        pp('dg_dy * df_dx', dg_dy * df_dx)
        self.assertEqual(dg_dx, dg_dy * df_dx)

    def test_chain_rule_rowwise(self):
        x = matrix('x', 1, 3)
        y = matrix('y', 1, 3)
        c = Matrix([[2, 3, 4]])
        f_x = x * Diag(c) + Matrix([[2 * x[0, 1], x[0, 1], x[0, 0]]])
        g_y = y * y.T
        g_fx = substitute(g_y, (y, f_x))

        pp('x', x)
        pp('f(x)', f_x)
        pp('g(f(x))', g_fx)
        pp('g(y)', g_y)
        pp('f(x).jacobian(x)', f_x.jacobian(x))
        pp('g(f(x)).jacobian(x)', g_fx.jacobian(x))
        pp('g(y).jacobian(y)', g_y.jacobian(y))

        df_dx = f_x.jacobian(x)
        dg_dy = substitute(g_y.jacobian(y), (y, f_x))
        dg_dx = g_fx.jacobian(x)

        pp('dg_dx', dg_dx)
        pp('dg_dy * df_dx', dg_dy * df_dx)
        self.assertEqual(dg_dx, dg_dy * df_dx)

    def test_lemma_fx_x_colwise(self):
        x = matrix('x', 3, 1)
        f = lambda x: 3 * x[0, 0] + 2 * x[1, 0] + 5 * x[2, 0]
        g = lambda x: f(x) * x
        f_x = f(x)
        g_x = g(x)

        pp('x', x)
        print('f(x)', f_x)
        pp('g(x)', g_x)

        df_dx = sp.Matrix([[f_x]]).jacobian(x)
        dg_dx = g_x.jacobian(x)

        pp('df_dx', df_dx)
        pp('dg_dx', dg_dx)
        self.assertEqual(dg_dx, x * df_dx + f_x * sp.eye(3))

    def test_lemma_fx_x_rowwise(self):
        x = matrix('x', 1, 3)
        f = lambda x: 3 * x[0, 0] + 2 * x[0, 1] + 5 * x[0, 2]
        g = lambda x: f(x) * x
        f_x = f(x)
        g_x = g(x)

        pp('x', x)
        print('f(x)', f_x)
        pp('g(x)', g_x)

        df_dx = sp.Matrix([[f_x]]).jacobian(x)
        dg_dx = g_x.jacobian(x)

        pp('df_dx', df_dx)
        pp('dg_dx', dg_dx)
        self.assertEqual(dg_dx, x.T * df_dx + f_x * sp.eye(3))


if __name__ == '__main__':
    import unittest
    unittest.main()
