#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

# see also https://docs.sympy.org/latest/modules/matrices/matrices.html

from typing import List
from unittest import TestCase

import sympy as sp
from sympy import Matrix

#-------------------------------------#
#           matrix operations
#-------------------------------------#

def is_column_vector(x: Matrix) -> bool:
    m, n = x.shape
    return n == 1


def is_row_vector(x: Matrix) -> bool:
    m, n = x.shape
    return m == 1


def is_square(X: Matrix) -> bool:
    m, n = X.shape
    return m == n


def join_columns(columns: List[Matrix]) -> Matrix:
    assert all(is_column_vector(column) for column in columns)
    return Matrix([x.T for x in columns]).T


def join_rows(rows: List[Matrix]) -> Matrix:
    assert all(is_row_vector(row) for row in rows)
    return Matrix(rows)


def diff(f, X: Matrix) -> Matrix:
    """
    Returns the derivative of a matrix function
    :param f: a real-valued function
    :param X: a matrix
    :return: the derivative of f
    """
    m, n = X.shape
    return Matrix([[sp.diff(f, X[i, j]) for j in range(n)] for i in range(m)])


def substitute(expr, X: Matrix, Y: Matrix):
    assert X.shape == Y.shape
    m, n = X.shape
    substitutions = ((X[i, j], Y[i, j]) for i in range(m) for j in range(n))
    return expr.subs(substitutions)


def jacobian(x: Matrix, y) -> Matrix:
    assert is_column_vector(x) or is_row_vector(x)
    return x.jacobian(y)


def exp(x: Matrix) -> Matrix:
    return x.applyfunc(lambda x: sp.exp(x))


def log(x: Matrix) -> Matrix:
    return x.applyfunc(lambda x: sp.log(x))


def inverse(x: Matrix) -> Matrix:
    return x.applyfunc(lambda x: 1 / x)


def diag(X: Matrix) -> Matrix:
    assert is_square(X)
    m, n = X.shape
    return Matrix([[X[i, i] for i in range(m)]]).T


def Diag(x: Matrix) -> Matrix:
    assert is_column_vector(x) or is_row_vector(x)
    return sp.diag(*x)


def hadamard(x: Matrix, y: Matrix) -> Matrix:
    assert x.shape == y.shape
    m, n = x.shape
    return Matrix([[x[i, j] * y[i, j] for j in range(n)] for i in range(m)])
    # return matrix_multiply_elementwise(x, y)  ==> this may cause errors:
    #     def mul_elementwise(A, B):
    # >       assert A.domain == B.domain
    # E       AssertionError


def sum_columns(X: Matrix) -> Matrix:
    m, n = X.shape
    columns = [sum(X.col(j)) for j in range(n)]
    return Matrix(columns).T


def sum_rows(X: Matrix) -> Matrix:
    m, n = X.shape
    rows = [sum(X.row(i)) for i in range(m)]
    return Matrix(rows)


def repeat_column(x: Matrix, n: int) -> Matrix:
    assert is_column_vector(x)
    rows, cols = x.shape
    rows = [[x[i, 0]] * n for i in range(rows)]
    return Matrix(rows)


def repeat_row(x: Matrix, n: int) -> Matrix:
    assert is_row_vector(x)
    rows, cols = x.shape
    columns = [[x[0, j]] * n for j in range(cols)]
    return Matrix(columns).T


#-------------------------------------#
#           loss(z) functions
#-------------------------------------#

def squared_error(X: Matrix):
    m, n = X.shape

    def f(x: Matrix) -> float:
        return sp.sqrt(sum(xi * xi for xi in x))

    return sum(f(X.col(j)) for j in range(n))


def sum_elements(X: Matrix):
    m, n = X.shape

    return sum(X[i, j] for i in range(m) for j in range(n))


#-------------------------------------#
#           softmax colwise
#-------------------------------------#

def softmax_colwise(X: Matrix) -> Matrix:
    m, n = X.shape
    E = exp(X)
    return hadamard(E, repeat_row(inverse(sum_columns(E)), m))


def softmax_colwise1(X: Matrix) -> Matrix:
    m, n = X.shape

    def softmax(x):
        e = exp(x)
        return e / sum(e)

    return Matrix([softmax(X.col(j)).T for j in range(n)]).T


def stable_softmax_colwise(X: Matrix) -> Matrix:
    m, n = X.shape
    c = Matrix(sp.symarray('C', (1, n), real=True))
    E = exp(X - repeat_row(c, m))
    return hadamard(E, repeat_row(inverse(sum_columns(E)), m))


def softmax_colwise_derivative(x: Matrix) -> Matrix:
    assert is_column_vector(x)
    y = softmax_colwise1(x)
    return Diag(y) - y * y.T


def softmax_colwise_derivative1(x: Matrix) -> Matrix:
    return jacobian(softmax_colwise1(x), x)


#-------------------------------------#
#           log_softmax colwise
#-------------------------------------#

def log_softmax_colwise(X: Matrix) -> Matrix:
    m, n = X.shape
    return X - repeat_row(log(sum_columns(exp(X))), m)


def log_softmax_colwise1(X: Matrix) -> Matrix:
    return log(softmax_colwise(X))


def stable_log_softmax_colwise(X: Matrix) -> Matrix:
    m, n = X.shape
    c = Matrix(sp.symarray('C', (1, n), real=True))
    Y = X - repeat_row(c, m)
    return Y - repeat_row(log(sum_columns(exp(Y))), m)


def log_softmax_colwise_derivative(x: Matrix) -> Matrix:
    assert is_column_vector(x)
    m, n = x.shape
    return sp.eye(m) - repeat_row(softmax_colwise(x).T, m)


def log_softmax_colwise_derivative1(x: Matrix) -> Matrix:
    return jacobian(log_softmax_colwise(x), x)


#-------------------------------------#
#           softmax rowwise
#-------------------------------------#

def softmax_rowwise(X: Matrix) -> Matrix:
    m, n = X.shape
    E = exp(X)
    return hadamard(E, repeat_column(inverse(sum_rows(E)), n))


def softmax_rowwise1(X: Matrix) -> Matrix:
    m, n = X.shape

    def softmax(x):
        e = exp(x)
        return e / sum(e)

    return join_rows([softmax(X.row(i)) for i in range(m)])


def softmax_rowwise2(X: Matrix) -> Matrix:
    return softmax_colwise(X.T).T


def stable_softmax_rowwise(X: Matrix) -> Matrix:
    m, n = X.shape
    c = Matrix(sp.symarray('C', (m, 1), real=True))
    E = exp(X - repeat_column(c, n))
    return hadamard(E, repeat_column(inverse(sum_rows(E)), n))


def softmax_rowwise_derivative(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    y = softmax_rowwise(x)
    return Diag(y) - y.T * y


def softmax_rowwise_derivative1(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    return jacobian(softmax_rowwise(x), x)


def softmax_rowwise_derivative2(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    return softmax_colwise_derivative(x.T).T


#-------------------------------------#
#           log_softmax rowwise
#-------------------------------------#

def log_softmax_rowwise(X: Matrix) -> Matrix:
    m, n = X.shape
    return X - repeat_column(log(sum_rows(exp(X))), n)


def log_softmax_rowwise1(X: Matrix) -> Matrix:
    return log(softmax_rowwise(X))


def log_softmax_rowwise2(X: Matrix) -> Matrix:
    return log_softmax_colwise(X.T).T


def stable_log_softmax_rowwise(X: Matrix) -> Matrix:
    m, n = X.shape
    c = Matrix(sp.symarray('C', (m, 1), real=True))
    Y = X - repeat_column(c, n)
    return Y - repeat_column(log(sum_rows(exp(Y))), n)


def log_softmax_rowwise_derivative(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    m, n = x.shape
    return sp.eye(n) - repeat_row(softmax_rowwise(x), n)


def log_softmax_rowwise_derivative1(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    return jacobian(log_softmax_rowwise(x), x)


def log_softmax_rowwise_derivative2(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    return log_softmax_colwise_derivative(x.T)


class TestSoftmax(TestCase):
    def test_softmax_colwise(self):
        m = 3
        n = 2
        X = Matrix(sp.symarray('x', (m, n), real=True))

        y1 = softmax_colwise(X)
        y2 = softmax_colwise1(X)
        y3 = stable_softmax_colwise(X)
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(m, n))
        self.assertEqual(sp.simplify(y1 - y3), sp.zeros(m, n))

        y1 = log_softmax_colwise(X)
        y2 = log_softmax_colwise1(X)
        y3 = stable_log_softmax_colwise(X)
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(m, n))
        self.assertEqual(sp.simplify(y1 - y3), sp.zeros(m, n))

    def test_softmax_rowwise(self):
        m = 2
        n = 3
        X = Matrix(sp.symarray('x', (m, n), real=True))

        y1 = softmax_rowwise(X)
        y2 = softmax_rowwise1(X)
        y3 = softmax_rowwise2(X)
        y4 = stable_softmax_rowwise(X)
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(m, n))
        self.assertEqual(sp.simplify(y1 - y3), sp.zeros(m, n))
        self.assertEqual(sp.simplify(y1 - y4), sp.zeros(m, n))

        y1 = log_softmax_rowwise(X)
        y2 = log_softmax_rowwise1(X)
        y3 = log_softmax_rowwise2(X)
        y4 = stable_log_softmax_rowwise(X)
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(m, n))
        self.assertEqual(sp.simplify(y1 - y3), sp.zeros(m, n))
        self.assertEqual(sp.simplify(y1 - y4), sp.zeros(m, n))

    def test_softmax_colwise_derivative(self):
        x = Matrix(sp.symbols('x y z'), real=True)
        m, n = x.shape

        y1 = sp.simplify(softmax_colwise_derivative(x))
        y2 = sp.simplify(softmax_colwise_derivative1(x))
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(m, m))

    def test_log_softmax_colwise_derivative(self):
        x = Matrix(sp.symbols('x y z'), real=True)
        m, n = x.shape

        y1 = sp.simplify(log_softmax_colwise_derivative(x))
        y2 = sp.simplify(log_softmax_colwise_derivative1(x))
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(m, m))

    def test_softmax_rowwise_derivative(self):
        x = Matrix(sp.symbols('x y z'), real=True).T
        m, n = x.shape

        y1 = sp.simplify(softmax_rowwise_derivative(x))
        y2 = sp.simplify(softmax_rowwise_derivative1(x))
        y3 = sp.simplify(softmax_rowwise_derivative2(x))
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(n, n))
        self.assertEqual(sp.simplify(y1 - y3), sp.zeros(n, n))

    def test_log_softmax_rowwise_derivative(self):
        x = Matrix(sp.symbols('x y z'), real=True).T
        m, n = x.shape

        y1 = sp.simplify(log_softmax_rowwise_derivative(x))
        y2 = sp.simplify(log_softmax_rowwise_derivative1(x))
        y3 = sp.simplify(log_softmax_rowwise_derivative2(x))
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(n, n))
        self.assertEqual(sp.simplify(y1 - y3), sp.zeros(n, n))


class TestMatrixOperations(TestCase):
    def test_matrix_operations(self):
        A = Matrix([[1, 2, 3]])
        B = repeat_row(A, 2)
        C = Matrix([[1, 2, 3],
                    [1, 2, 3]])
        self.assertEqual(B, C)


class TestSoftmaxLayers(TestCase):
    def test_softmax_layer_colwise(self):
        m = 3
        n = 2
        loss = squared_error

        y = Matrix(sp.symarray('y', (m, n), real=True))
        z = Matrix(sp.symarray('z', (m, n), real=True))

        Y = softmax_colwise(z)
        DY = substitute(diff(loss(y), y), y, Y)
        DZ1 = hadamard(Y, DY - repeat_row(diag(Y.T * DY).T, m))
        DZ2 = diff(loss(Y), z)

        self.assertEqual(sp.simplify(DZ1 - DZ2), sp.zeros(m, n))

    def test_log_softmax_layer_colwise(self):
        m = 3
        n = 2
        loss = squared_error

        y = Matrix(sp.symarray('y', (m, n), real=True))
        z = Matrix(sp.symarray('z', (m, n), real=True))

        Y = log_softmax_colwise(z)
        DY = substitute(diff(loss(y), y), y, Y)
        DZ1 = DY - hadamard(softmax_colwise(z), repeat_row(sum_columns(DY), m))
        DZ2 = diff(loss(Y), z)

        self.assertEqual(sp.simplify(DZ1 - DZ2), sp.zeros(m, n))

    def test_softmax_layer_rowwise(self):
        m = 2
        n = 3
        loss = squared_error

        y = Matrix(sp.symarray('y', (m, n), real=True))
        z = Matrix(sp.symarray('z', (m, n), real=True))

        Y = softmax_rowwise(z)
        DY = substitute(diff(loss(y), y), y, Y)
        DZ1 = hadamard(Y, DY - repeat_column(diag(DY * Y.T), n))
        DZ2 = diff(loss(Y), z)

        self.assertEqual(sp.simplify(DZ1 - DZ2), sp.zeros(m, n))

    def test_log_softmax_layer_rowwise(self):
        m = 2
        n = 3
        loss = squared_error

        y = Matrix(sp.symarray('y', (m, n), real=True))
        z = Matrix(sp.symarray('z', (m, n), real=True))

        Y = log_softmax_rowwise(z)
        DY = substitute(diff(loss(y), y), y, Y)
        DZ1 = DY - hadamard(softmax_rowwise(z), repeat_column(sum_rows(DY), n))
        DZ2 = diff(loss(Y), z)

        self.assertEqual(sp.simplify(DZ1 - DZ2), sp.zeros(m, n))


class TestBatchNormalization(TestCase):
    def test_batchnorm_colwise(self):
        D = 1
        N = 2
        loss = sum_elements

        x = Matrix(sp.symarray('x', (D, N), real=True))
        y = Matrix(sp.symarray('y', (D, N), real=True))

        X = x
        mu = sum_columns(X) / N
        R = X - repeat_row(mu, D)
        Sigma = diag(R * R.T) / N
        Sigma_power_minus_half = Sigma.applyfunc(lambda t: 1 / sp.sqrt(t))
        Y = hadamard(R, repeat_column(Sigma_power_minus_half, N))

        DY = substitute(diff(loss(y), y), y, Y)
        DX1 = hadamard(repeat_column(Sigma_power_minus_half, N) / N, -hadamard(Y, repeat_column(diag(DY * Y.T), N)) + DY * (N * sp.eye(N) - sp.ones(N, N)))
        DX2 = diff(loss(Y), x)

        x_ = Matrix([[7, 19]])
        d1 = sp.simplify(substitute(DX1, x, x_))
        d2 = sp.simplify(substitute(DX2, x, x_))
        print(d1)
        print(d2)
        #self.assertEqual(d1, d2)
        #self.assertEqual(sp.simplify(DX1 - DX2), sp.zeros(D, N))


class TestLemmas(TestCase):
    def test_lemma_colwise1(self):
        m = 2
        n = 3

        X = Matrix(sp.symarray('x', (m, n), real=True))
        Y = Matrix(sp.symarray('y', (m, n), real=True))
        Z1 = join_columns([X.col(j) * X.col(j).T * Y.col(j) for j in range(n)])
        Z2 = hadamard(X, repeat_row(diag(X.T * Y).T, m))
        self.assertEqual(sp.simplify(Z1 - Z2), sp.zeros(m, n))

    def test_lemma_colwise2(self):
        m = 2
        n = 3

        X = Matrix(sp.symarray('x', (m, n), real=True))
        Y = Matrix(sp.symarray('y', (m, n), real=True))
        Z1 = join_columns([repeat_row(X.col(j).T, m) * Y.col(j) for j in range(n)])
        Z2 = repeat_row(diag(X.T * Y).T, m)
        self.assertEqual(sp.simplify(Z1 - Z2), sp.zeros(m, n))

    def test_lemma_colwise3(self):
        m = 2
        n = 3

        X = Matrix(sp.symarray('x', (m, n), real=True))
        Y = Matrix(sp.symarray('y', (m, n), real=True))
        Z1 = join_columns([repeat_column(X.col(j), m) * Y.col(j) for j in range(n)])
        Z2 = hadamard(X, repeat_row(sum_columns(Y), m))
        self.assertEqual(sp.simplify(Z1 - Z2), sp.zeros(m, n))

    def test_lemma_rowwise1(self):
        m = 2
        n = 3

        X = Matrix(sp.symarray('x', (m, n), real=True))
        Y = Matrix(sp.symarray('y', (m, n), real=True))
        Z1 = join_rows([X.row(i) * Y.row(i).T * Y.row(i) for i in range(m)])
        Z2 = hadamard(Y, repeat_column(diag(X * Y.T), n))
        self.assertEqual(sp.simplify(Z1 - Z2), sp.zeros(m, n))

    def test_lemma_rowwise2(self):
        m = 2
        n = 3

        X = Matrix(sp.symarray('x', (m, n), real=True))
        Y = Matrix(sp.symarray('y', (m, n), real=True))
        Z1 = join_rows([X.row(i) * repeat_column(Y.row(i).T, n) for i in range(m)])
        Z2 = repeat_column(diag(X * Y.T), n)
        self.assertEqual(sp.simplify(Z1 - Z2), sp.zeros(m, n))

    def test_lemma_rowwise3(self):
        m = 2
        n = 3

        X = Matrix(sp.symarray('x', (m, n), real=True))
        Y = Matrix(sp.symarray('y', (m, n), real=True))
        Z1 = join_rows([X.row(i) * repeat_row(Y.row(i), n) for i in range(m)])
        Z2 = hadamard(Y, repeat_column(sum_rows(X), n))
        self.assertEqual(sp.simplify(Z1 - Z2), sp.zeros(m, n))


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
