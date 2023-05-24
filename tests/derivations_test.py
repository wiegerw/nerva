#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

# see also https://docs.sympy.org/latest/modules/matrices/matrices.html

from typing import List
from unittest import TestCase

import sympy as sp
from sympy import Matrix, matrix_multiply_elementwise

#-------------------------------------#
#           matrix functions
#-------------------------------------#

def join_columns(columns: List[Matrix]) -> Matrix:
    return Matrix([x.T for x in columns]).T


def diff(f, X: Matrix):
    m, n = X.shape
    return Matrix([[sp.diff(f, X[i, j]) for j in range(n)] for i in range(m)])


def substitute(expr, X: Matrix, Y: Matrix):
    m, n = X.shape
    assert m, n == Y.shape
    substitutions = ((X[i, j], Y[i, j]) for i in range(m) for j in range(n))
    return expr.subs(substitutions)


def jacobian(X: Matrix, Y: Matrix) -> Matrix:
    m, n = X.shape
    assert m == 1 or n == 1
    if n == 1:
        return X.jacobian(Y)
    else:
        return X.jacobian(Y).T


def exp(x: Matrix) -> Matrix:
    return x.applyfunc(lambda x: sp.exp(x))


def log(x: Matrix) -> Matrix:
    return x.applyfunc(lambda x: sp.log(x))


def inverse(x: Matrix) -> Matrix:
    return x.applyfunc(lambda x: 1 / x)


def diag(X: Matrix) -> Matrix:
    m, n = X.shape
    assert m == n
    return Matrix([[X[i, i] for i in range(m)]]).T


def Diag(x: Matrix) -> Matrix:
    m, n = x.shape
    assert m == 1 or n == 1
    return sp.diag(*x)


def hadamard(x: Matrix, y: Matrix) -> Matrix:
    return matrix_multiply_elementwise(x, y)


def colwise_sum(X: Matrix) -> Matrix:
    m, n = X.shape
    columns = [sum(X[:, j]) for j in range(n)]
    return Matrix(columns).T


def rowwise_sum(X: Matrix) -> Matrix:
    m, n = X.shape
    rows = [sum(X[i, :]) for i in range(m)]
    return Matrix(rows)


def rowwise_replicate(X: Matrix, n: int) -> Matrix:
    rows, cols = X.shape
    assert cols == 1
    rows = [[X[i, 0]] * n for i in range(rows)]
    return Matrix(rows)


def colwise_replicate(X: Matrix, n: int) -> Matrix:
    rows, cols = X.shape
    assert rows == 1
    columns = [[X[0, j]] * n for j in range(cols)]
    return Matrix(columns).T


#-------------------------------------#
#           loss functions
#-------------------------------------#

def squared_error(X: Matrix) -> float:
    m, n = X.shape

    def f(x: Matrix) -> float:
        return sp.sqrt(sum(xi * xi for xi in x))

    return sum(f(X[:, j]) for j in range(n))


#-------------------------------------#
#           softmax colwise
#-------------------------------------#

def softmax_colwise1(X: Matrix) -> Matrix:
    m, n = X.shape

    def softmax(x):
        e = exp(x)
        return e / sum(e)

    return Matrix([softmax(X[:, j]).T for j in range(n)]).T


def softmax_colwise2(X: Matrix) -> Matrix:
    m, n = X.shape
    E = exp(X)
    return hadamard(E, colwise_replicate(inverse(colwise_sum(E)), m))


# stable softmax
def softmax_colwise3(X: Matrix) -> Matrix:
    m, n = X.shape
    c = Matrix(sp.symarray('C', (1, n), real=True))
    E = exp(X - colwise_replicate(c, m))
    return hadamard(E, colwise_replicate(inverse(colwise_sum(E)), m))


def softmax_colwise_derivative1(x: Matrix) -> Matrix:
    return jacobian(softmax_colwise1(x), x)


def softmax_colwise_derivative2(x: Matrix) -> Matrix:
    m, n = x.shape
    assert n == 1
    y = softmax_colwise1(x)
    return Diag(y) - y * y.T


#-------------------------------------#
#           log_softmax colwise
#-------------------------------------#

def log_softmax_colwise1(X: Matrix) -> Matrix:
    return log(softmax_colwise1(X))


def log_softmax_colwise2(X: Matrix) -> Matrix:
    m, n = X.shape
    return X - colwise_replicate(log(colwise_sum(exp(X))), m)


def log_softmax_colwise3(X: Matrix) -> Matrix:
    m, n = X.shape
    c = Matrix(sp.symarray('C', (1, n), real=True))
    Y = X - colwise_replicate(c, m)
    return Y - colwise_replicate(log(colwise_sum(exp(Y))), m)


def log_softmax_colwise_derivative1(x: Matrix) -> Matrix:
    return jacobian(log_softmax_colwise1(x), x)


def log_softmax_colwise_derivative2(x: Matrix) -> Matrix:
    m, n = x.shape
    assert n == 1
    return sp.eye(m) - colwise_replicate(softmax_colwise2(x).T, m)


#-------------------------------------#
#           softmax rowwise
#-------------------------------------#

def softmax_rowwise1(X: Matrix) -> Matrix:
    m, n = X.shape

    def softmax(x):
        e = exp(x)
        return e / sum(e)

    return Matrix([softmax(X[i, :]) for i in range(m)])


def softmax_rowwise2(X: Matrix) -> Matrix:
    m, n = X.shape
    E = exp(X)
    return hadamard(E, rowwise_replicate(inverse(rowwise_sum(E)), n))


# stable softmax
def softmax_rowwise3(X: Matrix) -> Matrix:
    m, n = X.shape
    c = Matrix(sp.symarray('C', (m, 1), real=True))
    E = exp(X - rowwise_replicate(c, n))
    return hadamard(E, rowwise_replicate(inverse(rowwise_sum(E)), n))


def softmax_rowwise_derivative1(x: Matrix) -> Matrix:
    return jacobian(softmax_rowwise1(x), x)


def softmax_rowwise_derivative2(x: Matrix) -> Matrix:
    return softmax_colwise_derivative1(x.T).T


def softmax_rowwise_derivative3(x: Matrix) -> Matrix:
    m, n = x.shape
    assert m == 1
    y = softmax_rowwise1(x)
    return Diag(y) - y.T * y


#-------------------------------------#
#           log_softmax rowwise
#-------------------------------------#

def log_softmax_rowwise1(X: Matrix) -> Matrix:
    return log(softmax_rowwise1(X))


def log_softmax_rowwise2(X: Matrix) -> Matrix:
    m, n = X.shape
    return X - rowwise_replicate(log(rowwise_sum(exp(X))), n)


def log_softmax_rowwise3(X: Matrix) -> Matrix:
    m, n = X.shape
    c = Matrix(sp.symarray('C', (m, 1), real=True))
    Y = X - rowwise_replicate(c, n)
    return Y - rowwise_replicate(log(rowwise_sum(exp(Y))), n)


def log_softmax_rowwise_derivative1(x: Matrix) -> Matrix:
    return jacobian(log_softmax_rowwise1(x), x)


def log_softmax_rowwise_derivative2(x: Matrix) -> Matrix:
    return log_softmax_colwise_derivative1(x.T).T


def log_softmax_rowwise_derivative3(x: Matrix) -> Matrix:
    m, n = x.shape
    assert m == 1
    return sp.eye(n) - rowwise_replicate(softmax_rowwise2(x).T, n)

class TestDerivations(TestCase):
    def test_softmax_colwise(self):
        m = 3
        n = 2
        X = Matrix(sp.symarray('X', (m, n), real=True))

        y1 = softmax_colwise1(X)
        y2 = softmax_colwise2(X)
        y3 = softmax_colwise3(X)
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(m, n))
        self.assertEqual(sp.simplify(y1 - y3), sp.zeros(m, n))

        y1 = log_softmax_colwise1(X)
        y2 = log_softmax_colwise2(X)
        y3 = log_softmax_colwise3(X)
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(m, n))
        self.assertEqual(sp.simplify(y1 - y3), sp.zeros(m, n))

    def test_softmax_rowwise(self):
        m = 2
        n = 3
        X = Matrix(sp.symarray('X', (m, n), real=True))

        y1 = softmax_rowwise1(X)
        y2 = softmax_rowwise2(X)
        y3 = softmax_rowwise3(X)
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(m, n))
        self.assertEqual(sp.simplify(y1 - y3), sp.zeros(m, n))

        y1 = log_softmax_rowwise1(X)
        y2 = log_softmax_rowwise2(X)
        y3 = log_softmax_rowwise3(X)
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(m, n))
        self.assertEqual(sp.simplify(y1 - y3), sp.zeros(m, n))

    def test_softmax_colwise_derivative(self):
        x = Matrix(sp.symbols('x y z'), real=True)
        m, n = x.shape

        y1 = sp.simplify(softmax_colwise_derivative1(x))
        y2 = sp.simplify(softmax_colwise_derivative2(x))
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(m, m))

    def test_log_softmax_colwise_derivative(self):
        x = Matrix(sp.symbols('x y z'), real=True)
        m, n = x.shape

        y1 = sp.simplify(log_softmax_colwise_derivative1(x))
        y2 = sp.simplify(log_softmax_colwise_derivative2(x))
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(m, m))

    def test_softmax_rowwise_derivative(self):
        x = Matrix(sp.symbols('x y z'), real=True).T
        m, n = x.shape

        y1 = sp.simplify(softmax_rowwise_derivative1(x))
        y2 = sp.simplify(softmax_rowwise_derivative2(x))
        y3 = sp.simplify(softmax_rowwise_derivative3(x))
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(n, n))
        self.assertEqual(sp.simplify(y1 - y3), sp.zeros(n, n))

    def test_log_softmax_rowwise_derivative(self):
        x = Matrix(sp.symbols('x y z'), real=True).T
        m, n = x.shape

        y1 = sp.simplify(log_softmax_rowwise_derivative1(x))
        y2 = sp.simplify(log_softmax_rowwise_derivative2(x))
        y3 = sp.simplify(log_softmax_rowwise_derivative3(x))
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(n, n))
        self.assertEqual(sp.simplify(y1 - y3), sp.zeros(n, n))

    def test_softmax_layer_colwise(self):
        m = 3
        n = 2

        Y = Matrix(sp.symarray('Y', (m, n), real=True))
        Z = Matrix(sp.symarray('Z', (m, n), real=True))

        Y_Z = softmax_colwise1(Z)

        f_Y = squared_error(Y)
        f_Z = squared_error(Y_Z)

        DY_Z = substitute(diff(f_Y, Y), Y, Y_Z)
        DZ1 = diff(f_Z, Z)
        DZ2 = hadamard(Y_Z, DY_Z - colwise_replicate(diag(Y_Z.T * DY_Z).T, m))

        self.assertEqual(sp.simplify(DZ1 - DZ2), sp.zeros(m, n))

    def test_log_softmax_layer_colwise(self):
        m = 3
        n = 2

        Y = Matrix(sp.symarray('Y', (m, n), real=True))
        Z = Matrix(sp.symarray('Z', (m, n), real=True))

        Y_Z = log_softmax_colwise1(Z)

        f_Y = squared_error(Y)
        f_Z = squared_error(Y_Z)

        DY = diff(f_Y, Y)
        DY_Z = substitute(DY, Y, Y_Z)
        DZ1 = diff(f_Z, Z)
        DZ2 = DY_Z - hadamard(softmax_colwise1(Z), colwise_replicate(colwise_sum(DY_Z), m))

        self.assertEqual(sp.simplify(DZ1 - DZ2), sp.zeros(m, n))

    def test_softmax_layer_rowwise(self):
        m = 2
        n = 3

        Y = Matrix(sp.symarray('Y', (m, n), real=True))
        Z = Matrix(sp.symarray('Z', (m, n), real=True))

        Y_Z = softmax_rowwise1(Z)

        f_Y = squared_error(Y)
        f_Z = squared_error(Y_Z)

        DY_Z = substitute(diff(f_Y, Y), Y, Y_Z)
        DZ1 = diff(f_Z, Z)
        DZ2 = hadamard(Y_Z, DY_Z - rowwise_replicate(diag(DY_Z * Y_Z.T), n))

        self.assertEqual(sp.simplify(DZ1 - DZ2), sp.zeros(m, n))

    def test_matrix_operations(self):
        A = Matrix([[1, 2, 3]])
        B = colwise_replicate(A, 2)
        C = Matrix([[1, 2, 3],
                    [1, 2, 3]])
        self.assertEqual(B, C)


class TestLemmas(TestCase):
    def test_lemma1(self):
        m = 2
        n = 3

        X = Matrix(sp.symarray('X', (m, n), real=True))
        Y = Matrix(sp.symarray('Y', (m, n), real=True))
        Z1 = join_columns([X.col(j) * X.col(j).T * Y.col(j) for j in range(n)])
        Z2 = hadamard(X, colwise_replicate(diag(X.T * Y).T, m))
        self.assertEqual(sp.simplify(Z1 - Z2), sp.zeros(m, n))

    def test_lemma2(self):
        m = 2
        n = 3

        X = Matrix(sp.symarray('X', (m, n), real=True))
        Y = Matrix(sp.symarray('Y', (m, n), real=True))
        Z1 = join_columns([colwise_replicate(X.col(j).T, m) * Y.col(j) for j in range(n)])
        Z2 = colwise_replicate(diag(X.T * Y).T, m)
        self.assertEqual(sp.simplify(Z1 - Z2), sp.zeros(m, n))

    def test_lemma3(self):
        m = 2
        n = 3

        X = Matrix(sp.symarray('X', (m, n), real=True))
        Y = Matrix(sp.symarray('Y', (m, n), real=True))
        Z1 = Matrix([X.row(i) * Y.row(i).T * Y.row(i) for i in range(m)])
        Z2 = hadamard(Y, rowwise_replicate(diag(X * Y.T), n))
        self.assertEqual(sp.simplify(Z1 - Z2), sp.zeros(m, n))

    def test_lemma4(self):
        m = 2
        n = 3

        X = Matrix(sp.symarray('X', (m, n), real=True))
        Y = Matrix(sp.symarray('Y', (m, n), real=True))
        Z1 = Matrix([X.row(i) * rowwise_replicate(Y.row(i).T, n) for i in range(m)])
        Z2 = rowwise_replicate(diag(X * Y.T), n)
        self.assertEqual(sp.simplify(Z1 - Z2), sp.zeros(m, n))


if __name__ == '__main__':
    import unittest
    unittest.main()
