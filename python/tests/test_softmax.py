#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase
from symbolic.softmax_sympy import *

#-------------------------------------#
# alternative implementations of softmax functions
#-------------------------------------#

def softmax_colwise1(X: Matrix) -> Matrix:
    D, N = X.shape

    def softmax(x):
        e = exp(x)
        return e / sum(e)

    return Matrix([softmax(X.col(j)).T for j in range(N)]).T


def softmax_colwise_derivative1(x: Matrix) -> Matrix:
    return jacobian(softmax_colwise1(x), x)


def log_softmax_colwise1(X: Matrix) -> Matrix:
    return log(softmax_colwise(X))


def log_softmax_colwise_derivative1(x: Matrix) -> Matrix:
    return jacobian(log_softmax_colwise(x), x)


def softmax_rowwise1(X: Matrix) -> Matrix:
    N, D = X.shape

    def softmax(x):
        e = exp(x)
        return e / sum(e)

    return join_rows([softmax(X.row(i)) for i in range(N)])


def softmax_rowwise2(X: Matrix) -> Matrix:
    return softmax_colwise(X.T).T


def softmax_rowwise_derivative1(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    return jacobian(softmax_rowwise(x), x)


def softmax_rowwise_derivative2(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    return softmax_colwise_derivative(x.T).T


def log_softmax_rowwise1(X: Matrix) -> Matrix:
    return log(softmax_rowwise(X))


def log_softmax_rowwise2(X: Matrix) -> Matrix:
    return log_softmax_colwise(X.T).T


def log_softmax_rowwise_derivative1(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    return jacobian(log_softmax_rowwise(x), x)


def log_softmax_rowwise_derivative2(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    return log_softmax_colwise_derivative(x.T)


class TestSoftmax(TestCase):
    def test_softmax_colwise(self):
        D = 3
        N = 2
        X = Matrix(sp.symarray('x', (D, N), real=True))

        y1 = softmax_colwise(X)
        y2 = softmax_colwise1(X)
        y3 = stable_softmax_colwise(X)
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(D, N))
        self.assertEqual(sp.simplify(y1 - y3), sp.zeros(D, N))

        y1 = log_softmax_colwise(X)
        y2 = log_softmax_colwise1(X)
        y3 = stable_log_softmax_colwise(X)
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(D, N))
        self.assertEqual(sp.simplify(y1 - y3), sp.zeros(D, N))

    def test_softmax_colwise_derivative(self):
        x = Matrix(sp.symbols('x y z'), real=True)
        D, N = x.shape

        y1 = sp.simplify(softmax_colwise_derivative(x))
        y2 = sp.simplify(softmax_colwise_derivative1(x))
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(D, D))

    def test_log_softmax_colwise_derivative(self):
        x = Matrix(sp.symbols('x y z'), real=True)
        D, N = x.shape

        y1 = sp.simplify(log_softmax_colwise_derivative(x))
        y2 = sp.simplify(log_softmax_colwise_derivative1(x))
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(D, D))

    def test_softmax_rowwise(self):
        D = 3
        N = 2
        X = Matrix(sp.symarray('x', (N, D), real=True))

        y1 = softmax_rowwise(X)
        y2 = softmax_rowwise1(X)
        y3 = softmax_rowwise2(X)
        y4 = stable_softmax_rowwise(X)
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(N, D))
        self.assertEqual(sp.simplify(y1 - y3), sp.zeros(N, D))
        self.assertEqual(sp.simplify(y1 - y4), sp.zeros(N, D))

        y1 = log_softmax_rowwise(X)
        y2 = log_softmax_rowwise1(X)
        y3 = log_softmax_rowwise2(X)
        y4 = stable_log_softmax_rowwise(X)
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(N, D))
        self.assertEqual(sp.simplify(y1 - y3), sp.zeros(N, D))
        self.assertEqual(sp.simplify(y1 - y4), sp.zeros(N, D))

    def test_softmax_rowwise_derivative(self):
        x = Matrix(sp.symbols('x y z'), real=True).T
        N, D = x.shape

        y1 = sp.simplify(softmax_rowwise_derivative(x))
        y2 = sp.simplify(softmax_rowwise_derivative1(x))
        y3 = sp.simplify(softmax_rowwise_derivative2(x))
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(D, D))
        self.assertEqual(sp.simplify(y1 - y3), sp.zeros(D, D))

    def test_log_softmax_rowwise_derivative(self):
        x = Matrix(sp.symbols('x y z'), real=True).T
        N, D = x.shape

        y1 = sp.simplify(log_softmax_rowwise_derivative(x))
        y2 = sp.simplify(log_softmax_rowwise_derivative1(x))
        y3 = sp.simplify(log_softmax_rowwise_derivative2(x))
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(D, D))
        self.assertEqual(sp.simplify(y1 - y3), sp.zeros(D, D))


if __name__ == '__main__':
    import unittest
    unittest.main()
