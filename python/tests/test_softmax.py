#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase
from symbolic.softmax import *


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


if __name__ == '__main__':
    import unittest
    unittest.main()
