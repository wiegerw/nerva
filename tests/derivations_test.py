#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

# see also https://docs.sympy.org/latest/modules/matrices/matrices.html

from unittest import TestCase

import sympy as sp
from sympy import Matrix, matrix_multiply_elementwise


def exp(x: Matrix) -> Matrix:
    return x.applyfunc(lambda x: sp.exp(x))


def log(x: Matrix) -> Matrix:
    return x.applyfunc(lambda x: sp.log(x))


def div(x: Matrix) -> Matrix:
    return x.applyfunc(lambda x: 1 / x)


def diag(X: Matrix) -> Matrix:
    m, n = X.shape
    assert m == n
    return Matrix([[X[i, i] for i in range(m)]])


def Diag(x: Matrix) -> Matrix:
    m, n = x.shape
    assert n == 1
    return sp.diag(*x)


def diff(f, X: Matrix):
    m, n = X.shape
    return Matrix([[sp.diff(f, X[i, j]) for j in range(n)] for i in range(m)])


def substitute(expr, X: Matrix, Y: Matrix):
    m, n = X.shape
    assert m, n == Y.shape
    substitutions = ((X[i, j], Y[i, j]) for i in range(m) for j in range(n))
    return expr.subs(substitutions)


def hadamard(x: Matrix, y: Matrix) -> Matrix:
    return matrix_multiply_elementwise(x, y)


def colwise_sum(X: Matrix) -> Matrix:
    rows, cols = X.shape

    columns = [sum(X[:, j]) for j in range(cols)]
    return Matrix(columns).T


def replicate_rowwise(X: Matrix, n: int) -> Matrix:
    rows, cols = X.shape
    assert rows == 1
    columns = [[X[j]] * n for j in range(cols)]
    return Matrix(columns).T


def softmax_colwise1(X: Matrix) -> Matrix:
    m, n = X.shape

    def softmax(x):
        e = exp(x)
        return e / sum(e)

    return Matrix([softmax(X[:, j]).T for j in range(n)]).T


def softmax_colwise2(X: Matrix) -> Matrix:
    m, n = X.shape
    E = exp(X)
    return hadamard(E, replicate_rowwise(div(colwise_sum(E)), m))


# stable softmax
def softmax_colwise3(X: Matrix) -> Matrix:
    m, n = X.shape
    c = Matrix(sp.symarray('C', (1, n), real=True))
    E = exp(X - replicate_rowwise(c, m))
    return hadamard(E, replicate_rowwise(div(colwise_sum(E)), m))


def log_softmax_colwise1(X: Matrix) -> Matrix:
    return log(softmax_colwise1(X))


def log_softmax_colwise2(X: Matrix) -> Matrix:
    m, n = X.shape
    return X - replicate_rowwise(log(colwise_sum(exp(X))), m)


def log_softmax_colwise3(X: Matrix) -> Matrix:
    m, n = X.shape
    c = Matrix(sp.symarray('C', (1, n), real=True))
    Y = X - replicate_rowwise(c, m)
    return Y - replicate_rowwise(log(colwise_sum(exp(Y))), m)


def derivative_softmax_colwise1(x: Matrix) -> Matrix:
    return softmax_colwise1(x).jacobian(x)


def derivative_softmax_colwise2(x: Matrix) -> Matrix:
    m, n = x.shape
    assert n == 1
    y = softmax_colwise1(x)
    return Diag(y) - y * y.T


def derivative_log_softmax_colwise1(x: Matrix) -> Matrix:
    return log_softmax_colwise1(x).jacobian(x)


def derivative_log_softmax_colwise2(x: Matrix) -> Matrix:
    m, n = x.shape
    assert n == 1
    return sp.eye(m) - replicate_rowwise(softmax_colwise2(x).T, m)


def squared_error(X: Matrix) -> float:
    m, n = X.shape

    def f(x: Matrix) -> float:
        return sp.sqrt(sum(xi * xi for xi in x))

    return sum(f(X[:, j]) for j in range(n))


def run3():
    from sympy import sin, cos, Matrix
    from sympy.abc import rho, phi
    X = Matrix([rho * cos(phi), rho * sin(phi), rho ** 2])
    Y = Matrix([rho, phi])
    print(X)
    print(X.T)
    print(X.jacobian(Y))
    print((X.T).jacobian(Y))


class TestDerivations(TestCase):
    def test_softmax(self):
        m = 3
        n = 2
        X = Matrix(sp.symarray('X', (m, n), real=True))

        softmax1 = softmax_colwise1(X)
        softmax2 = softmax_colwise2(X)
        softmax3 = softmax_colwise3(X)
        log_softmax1 = log_softmax_colwise1(X)
        log_softmax2 = log_softmax_colwise2(X)
        log_softmax3 = log_softmax_colwise3(X)

        u = sp.simplify(softmax1 - softmax2)
        self.assertEqual(u, sp.zeros(m, n))

        u = sp.simplify(softmax1 - softmax3)
        self.assertEqual(u, sp.zeros(m, n))

        u = sp.simplify(log_softmax1 - log_softmax2)
        self.assertEqual(u, sp.zeros(m, n))

        u = sp.simplify(log_softmax1 - log_softmax3)
        self.assertEqual(u, sp.zeros(m, n))

    def test_softmax_derivative(self):
        x = Matrix(sp.symbols('x y z'), real=True)
        m, n = x.shape

        deriv_softmax1 = sp.simplify(derivative_softmax_colwise1(x))
        deriv_softmax2 = sp.simplify(derivative_softmax_colwise2(x))

        u = sp.simplify(deriv_softmax1 - deriv_softmax2)
        self.assertEqual(u, sp.zeros(m, m))

    def test_log_softmax_derivative(self):
        x = Matrix(sp.symbols('x y z'), real=True)
        m, n = x.shape

        deriv_softmax1 = sp.simplify(derivative_log_softmax_colwise1(x))
        deriv_softmax2 = sp.simplify(derivative_log_softmax_colwise2(x))

        u = sp.simplify(deriv_softmax1 - deriv_softmax2)
        self.assertEqual(u, sp.zeros(m, m))

    def test_softmax_layer(self):
        m = 3
        n = 2

        Y = Matrix(sp.symarray('Y', (m, n), real=True))
        Z = Matrix(sp.symarray('Z', (m, n), real=True))

        Y_Z = softmax_colwise1(Z)

        fY = squared_error(Y)
        fZ = squared_error(Y_Z)

        DY_Z = substitute(diff(fY, Y), Y, Y_Z)
        DZ = diff(fZ, Z)
        DZ1 = hadamard(Y_Z, DY_Z - replicate_rowwise(diag(Y_Z.T * DY_Z), m))

        u = sp.simplify(DZ - DZ1)
        self.assertEqual(u, sp.zeros(m, n))

    def log_softmax_layer_test(self):
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
        DZ2 = DY_Z - hadamard(softmax_colwise1(Z), replicate_rowwise(colwise_sum(DY_Z), m))

        u = sp.simplify(DZ1 - DZ2)
        self.assertEqual(u, sp.zeros(m, n))


if __name__ == '__main__':
    import unittest
    unittest.main()
