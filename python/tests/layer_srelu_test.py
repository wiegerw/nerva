#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase
from symbolic.activation_functions import *
from symbolic.matrix_operations import *
from symbolic.utilities import *


class TestSReLULayers(TestCase):

    # test for both colwise and rowwise
    def test_srelu_layer(self):
        N = 2
        K = 2
        loss = sum_elements

        # variables
        y = matrix('y', K, N)
        z = matrix('z', K, N)
        al = sp.symbols('al', real=True)
        tl = sp.symbols('tl', real=True)
        ar = sp.symbols('ar', real=True)
        tr = sp.symbols('tr', real=True)

        act = srelu(al, tl, ar, tr)
        act_prime = srelu_prime(al, tl, ar, tr)

        # feedforward
        Z = z
        Y = apply(act, Z)

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DZ = hadamard(DY, apply(act_prime, Z))

        Zij = sp.symbols('Zij')
        Al = apply(Lambda(Zij, Piecewise((Zij - tl, Zij <= tl), (0, True))), Z)
        Ar = apply(Lambda(Zij, Piecewise((0, Zij <= tl), (0, Zij < tr), (Zij - tr, True))), Z)
        Tl = apply(Lambda(Zij, Piecewise((1 - al, Zij <= tl), (0, True))), Z)
        Tr = apply(Lambda(Zij, Piecewise((0, Zij <= tl), (0, Zij < tr), (1 - ar, True))), Z)

        Dal = Matrix([[sum_elements(hadamard(DY, Al))]])
        Dar = Matrix([[sum_elements(hadamard(DY, Ar))]])
        Dtl = Matrix([[sum_elements(hadamard(DY, Tl))]])
        Dtr = Matrix([[sum_elements(hadamard(DY, Tr))]])

        # symbolic differentiation
        DZ1 = diff(loss(Y), z)
        Dal1 = diff(loss(Y), Matrix([[al]]))
        Dtl1 = diff(loss(Y), Matrix([[tl]]))
        Dar1 = diff(loss(Y), Matrix([[ar]]))
        Dtr1 = diff(loss(Y), Matrix([[tr]]))

        self.assertTrue(equal_matrices(DZ, DZ1, simplify_arguments=True))
        self.assertTrue(equal_matrices(Dal, Dal1, simplify_arguments=True))
        self.assertTrue(equal_matrices(Dtl, Dtl1, simplify_arguments=True))
        self.assertTrue(equal_matrices(Dar, Dar1, simplify_arguments=False))
        self.assertTrue(equal_matrices(Dtr, Dtr1, simplify_arguments=True))


if __name__ == '__main__':
    import unittest
    unittest.main()
