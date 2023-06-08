#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase
from symbolic.activation_functions_sympy_1d import *


class TestActivationFunctions1D(TestCase):

    def test_relu(self):
        f = relu
        f1 = relu_prime
        x = sp.symbols('x', real=True)
        self.assertEqual(sp.simplify(f1(x)), sp.simplify(f(x).diff(x)))

    def test_leaky_relu(self):
        alpha = sp.symbols('alpha', real=True)
        f = leaky_relu(alpha)
        f1 = leaky_relu_prime(alpha)
        x = sp.symbols('x', real=True)
        self.assertEqual(sp.simplify(f1(x)), sp.simplify(f(x).diff(x)))

    def test_all_relu(self):
        alpha = sp.symbols('alpha', real=True)
        f = all_relu(alpha)
        f1 = all_relu_prime(alpha)
        x = sp.symbols('x', real=True)
        self.assertEqual(sp.simplify(f1(x)), sp.simplify(f(x).diff(x)))

    def test_hyperbolic_tangent(self):
        f = hyperbolic_tangent
        f1 = hyperbolic_tangent_prime
        x = sp.symbols('x', real=True)
        self.assertEqual(sp.simplify(f1(x)), sp.simplify(f(x).diff(x)))

    def test_sigmoid(self):
        f = sigmoid
        f1 = sigmoid_prime
        x = sp.symbols('x', real=True)
        self.assertEqual(sp.simplify(f1(x)), sp.simplify(f(x).diff(x)))

    def test_srelu(self):
        al = sp.symbols('al', real=True)
        tl = sp.symbols('tl', real=True)
        ar = sp.symbols('ar', real=True)
        tr = sp.symbols('tr', real=True)

        f = srelu(al, tl, ar, tr)
        f1 = srelu_prime(al, tl, ar, tr)
        x = sp.symbols('x', real=True)
        self.assertEqual(f1(x), f(x).diff(x))


if __name__ == '__main__':
    import unittest
    unittest.main()
