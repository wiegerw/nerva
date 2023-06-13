#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase
from symbolic.activation_functions_sympy import *


class TestActivationFunctions1D(TestCase):

    def test_relu(self):
        f = relu_1d
        f1 = relu_1d_derivative
        x = sp.symbols('x', real=True)
        self.assertEqual(sp.simplify(f1(x)), sp.simplify(f(x).diff(x)))

    def test_leaky_relu(self):
        alpha = sp.symbols('alpha', real=True)
        f = leaky_relu_1d(alpha)
        f1 = leaky_relu_1d_derivative(alpha)
        x = sp.symbols('x', real=True)
        self.assertEqual(sp.simplify(f1(x)), sp.simplify(f(x).diff(x)))

    def test_all_relu(self):
        alpha = sp.symbols('alpha', real=True)
        f = all_relu_1d(alpha)
        f1 = all_relu_1d_derivative(alpha)
        x = sp.symbols('x', real=True)
        self.assertEqual(sp.simplify(f1(x)), sp.simplify(f(x).diff(x)))

    def test_hyperbolic_tangent(self):
        f = hyperbolic_tangent_1d
        f1 = hyperbolic_tangent_1d_derivative
        x = sp.symbols('x', real=True)
        self.assertEqual(sp.simplify(f1(x)), sp.simplify(f(x).diff(x)))

    def test_sigmoid(self):
        f = sigmoid_1d
        f1 = sigmoid_derivative_1d
        x = sp.symbols('x', real=True)
        self.assertEqual(sp.simplify(f1(x)), sp.simplify(f(x).diff(x)))

    def test_srelu(self):
        al = sp.symbols('al', real=True)
        tl = sp.symbols('tl', real=True)
        ar = sp.symbols('ar', real=True)
        tr = sp.symbols('tr', real=True)

        f = srelu_1d(al, tl, ar, tr)
        f1 = srelu_derivative_1d(al, tl, ar, tr)
        x = sp.symbols('x', real=True)
        self.assertEqual(f1(x), f(x).diff(x))


class TestLogSigmoid(TestCase):

    def test_log_sigmoid(self):
        f = lambda x: sp.log(sigmoid_1d(x))
        f1 = lambda x: 1 - sigmoid_1d(x)
        x = sp.symbols('x', real=True)
        self.assertEqual(sp.simplify(f1(x)), sp.simplify(f(x).diff(x)))


if __name__ == '__main__':
    import unittest
    unittest.main()
