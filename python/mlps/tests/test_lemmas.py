#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase

from mlps.nerva_sympy.matrix_operations import *


class TestLemmas(TestCase):
    def test_lemma_colwise1(self):
        m = 2
        n = 3

        X = Matrix(sp.symarray('x', (m, n), real=True))
        Y = Matrix(sp.symarray('y', (m, n), real=True))
        Z1 = join_columns([X.col(j) * X.col(j).T * Y.col(j) for j in range(n)])
        Z2 = hadamard(X, row_repeat(diag(X.T * Y).T, m))
        self.assertEqual(sp.simplify(Z1 - Z2), sp.zeros(m, n))

    def test_lemma_colwise2(self):
        m = 2
        n = 3

        X = Matrix(sp.symarray('x', (m, n), real=True))
        Y = Matrix(sp.symarray('y', (m, n), real=True))
        Z1 = join_columns([row_repeat(X.col(j).T, m) * Y.col(j) for j in range(n)])
        Z2 = row_repeat(diag(X.T * Y).T, m)
        self.assertEqual(sp.simplify(Z1 - Z2), sp.zeros(m, n))

    def test_lemma_colwise3(self):
        m = 2
        n = 3

        X = Matrix(sp.symarray('x', (m, n), real=True))
        Y = Matrix(sp.symarray('y', (m, n), real=True))
        Z1 = join_columns([column_repeat(X.col(j), m) * Y.col(j) for j in range(n)])
        Z2 = hadamard(X, row_repeat(columns_sum(Y), m))
        self.assertEqual(sp.simplify(Z1 - Z2), sp.zeros(m, n))

    def test_lemma_colwise4(self):
        m = 2
        n = 3

        X = Matrix(sp.symarray('x', (m, n), real=True))
        Y = Matrix(sp.symarray('y', (m, n), real=True))
        Z1 = sum(dot(X.col(j), Y.col(j)) for j in range(n))
        Z2 = elements_sum(hadamard(X, Y))
        self.assertEqual(Z1, Z2)

    def test_lemma_rowwise1(self):
        m = 2
        n = 3

        X = Matrix(sp.symarray('x', (m, n), real=True))
        Y = Matrix(sp.symarray('y', (m, n), real=True))
        Z1 = join_rows([X.row(i) * Y.row(i).T * Y.row(i) for i in range(m)])
        Z2 = hadamard(Y, column_repeat(diag(X * Y.T), n))
        self.assertEqual(sp.simplify(Z1 - Z2), sp.zeros(m, n))

    def test_lemma_rowwise2(self):
        m = 2
        n = 3

        X = Matrix(sp.symarray('x', (m, n), real=True))
        Y = Matrix(sp.symarray('y', (m, n), real=True))
        Z1 = join_rows([X.row(i) * column_repeat(Y.row(i).T, n) for i in range(m)])
        Z2 = column_repeat(diag(X * Y.T), n)
        self.assertEqual(sp.simplify(Z1 - Z2), sp.zeros(m, n))

    def test_lemma_rowwise3(self):
        m = 2
        n = 3

        X = Matrix(sp.symarray('x', (m, n), real=True))
        Y = Matrix(sp.symarray('y', (m, n), real=True))
        Z1 = join_rows([X.row(i) * row_repeat(Y.row(i), n) for i in range(m)])
        Z2 = hadamard(Y, column_repeat(rows_sum(X), n))
        self.assertEqual(sp.simplify(Z1 - Z2), sp.zeros(m, n))

    def test_lemma_rowwise4(self):
        m = 2
        n = 3

        X = Matrix(sp.symarray('x', (m, n), real=True))
        Y = Matrix(sp.symarray('y', (m, n), real=True))
        Z1 = sum(dot(X.row(i).T, Y.row(i).T) for i in range(m))
        Z2 = elements_sum(hadamard(X, Y))
        self.assertEqual(Z1, Z2)


class TestDerivatives(TestCase):
    def test_derivative_gx_x_colwise(self):
        n = 3
        x = Matrix(sp.symbols('x:{}'.format(n)))
        self.assertTrue(is_column_vector(x))

        g = sp.Function('g', real=True)(*x)
        J1 = jacobian(g * x, x)
        J2 = x * jacobian(Matrix([[g]]), x) + g * sp.eye(n)
        self.assertEqual(sp.simplify(J1 - J2), sp.zeros(n, n))

    def test_derivative_gx_x_rowwise(self):
        n = 3
        x = Matrix(sp.symbols('x:{}'.format(n))).T
        self.assertTrue(is_row_vector(x))

        g = sp.Function('g', real=True)(*x)
        J1 = jacobian(g * x, x)
        J2 = x.T * jacobian(Matrix([[g]]), x) + g * sp.eye(n)
        self.assertEqual(sp.simplify(J1 - J2), sp.zeros(n, n))


if __name__ == '__main__':
    import unittest
    unittest.main()
