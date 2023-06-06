#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase
from symbolic.activation_functions import *
from symbolic.matrix_operations import *
from symbolic.utilities import *


class TestSReLULayers(TestCase):

    def test_srelu_layer_colwise(self):
        D = 2
        K = 2
        N = 2
        loss = sum_elements

        # variables
        x = matrix('x', D, N)
        y = matrix('y', K, N)
        z = matrix('z', K, N)
        w = matrix('w', K, D)
        b = matrix('b', K, 1)
        al = sp.symbols('al', real=True)
        tl = sp.symbols('tl', real=True)
        ar = sp.symbols('ar', real=True)
        tr = sp.symbols('tr', real=True)

        act = srelu(al, tl, ar, tr)
        act_prime = srelu_prime(al, tl, ar, tr)

        # feedforward
        X = x
        W = w
        Z = W * X + repeat_column(b, N)
        Y = apply(act, Z)

        # helper functions
        Zij = sp.symbols('Zij')
        f_Al = Lambda(Zij, Piecewise((Zij - tl, Zij <= tl), (0, True)))
        f_Ar = Lambda(Zij, Piecewise((0, Zij <= tl), (0, Zij < tr), (Zij - tr, True)))
        f_Tl = Lambda(Zij, Piecewise((1 - al, Zij <= tl), (0, True)))
        f_Tr = Lambda(Zij, Piecewise((0, Zij <= tl), (0, Zij < tr), (1 - ar, True)))

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DZ = hadamard(DY, apply(act_prime, Z))
        DW = DZ * X.T
        Db = sum_rows(DZ)
        DX = W.T * DZ
        Al = apply(f_Al, Z)
        Ar = apply(f_Ar, Z)
        Tl = apply(f_Tl, Z)
        Tr = apply(f_Tr, Z)
        Dal = sum_elements(hadamard(DY, Al))
        Dar = sum_elements(hadamard(DY, Ar))
        Dtl = sum_elements(hadamard(DY, Tl))
        Dtr = sum_elements(hadamard(DY, Tr))

        # test gradients
        DZ1 = substitute(diff(loss(apply(act, z)), z), z, Z)
        DW1 = diff(loss(Y), w)
        Db1 = diff(loss(Y), b)
        DX1 = diff(loss(Y), x)
        self.assertTrue(equal_matrices(DZ, DZ1))
        self.assertTrue(equal_matrices(DW, DW1, simplify_arguments=True))
        self.assertTrue(equal_matrices(Db, Db1, simplify_arguments=True))
        self.assertTrue(equal_matrices(DX, DX1, simplify_arguments=True))

        Dal1 = diff(loss(Y), Matrix([[al]]))
        Dtl1 = diff(loss(Y), Matrix([[tl]]))
        Dar1 = diff(loss(Y), Matrix([[ar]]))
        Dtr1 = diff(loss(Y), Matrix([[tr]]))

        # wrap values in a matrix to make them usable for the equal_matrices function
        Dal = Matrix([[Dal]])
        Dar = Matrix([[Dar]])
        Dtl = Matrix([[Dtl]])
        Dtr = Matrix([[Dtr]])
        self.assertTrue(equal_matrices(Dal, Dal1, simplify_arguments=True))
        self.assertTrue(equal_matrices(Dtl, Dtl1, simplify_arguments=True))
        self.assertTrue(equal_matrices(Dar, Dar1, simplify_arguments=False))
        self.assertTrue(equal_matrices(Dtr, Dtr1, simplify_arguments=True))


    def test_srelu_layer_rowwise(self):
        D = 2
        K = 2
        N = 2
        loss = sum_elements

        # variables
        x = matrix('x', N, D)
        y = matrix('y', N, K)
        z = matrix('z', N, K)
        w = matrix('w', K, D)
        b = matrix('b', 1, K)
        al = sp.symbols('al', real=True)
        tl = sp.symbols('tl', real=True)
        ar = sp.symbols('ar', real=True)
        tr = sp.symbols('tr', real=True)

        act = srelu(al, tl, ar, tr)
        act_prime = srelu_prime(al, tl, ar, tr)

        # feedforward
        X = x
        W = w
        Z = X * W.T + repeat_row(b, N)
        Y = apply(act, Z)

        # helper functions
        Zij = sp.symbols('Zij')
        Al = Lambda(Zij, Piecewise((Zij - tl, Zij <= tl), (0, True)))
        Ar = Lambda(Zij, Piecewise((0, Zij <= tl), (0, Zij < tr), (Zij - tr, True)))
        Tl = Lambda(Zij, Piecewise((1 - al, Zij <= tl), (0, True)))
        Tr = Lambda(Zij, Piecewise((0, Zij <= tl), (0, Zij < tr), (1 - ar, True)))

        # backpropagation
        DY = substitute(diff(loss(y), y), y, Y)
        DZ = hadamard(DY, apply(act_prime, Z))
        DW = DZ.T * X
        Db = sum_columns(DZ)
        DX = DZ * W
        Dal = sum_elements(hadamard(DY, Al))
        Dar = sum_elements(hadamard(DY, Ar))
        Dtl = sum_elements(hadamard(DY, Tl))
        Dtr = sum_elements(hadamard(DY, Tr))
        Dal = sum_elements(apply(hadamard(DY, Al)))
        Dar = sum_elements(apply(hadamard(DY, Ar)))
        Dtl = sum_elements(apply(hadamard(DY, Tl)))
        Dtr = sum_elements(apply(hadamard(DY, Tr)))

        # test gradients
        DZ1 = substitute(diff(loss(apply(act, z)), z), z, Z)
        DW1 = diff(loss(Y), w)
        Db1 = diff(loss(Y), b)
        DX1 = diff(loss(Y), x)
        self.assertTrue(equal_matrices(DZ, DZ1))
        self.assertTrue(equal_matrices(DW, DW1, simplify_arguments=True))
        self.assertTrue(equal_matrices(Db, Db1, simplify_arguments=True))
        self.assertTrue(equal_matrices(DX, DX1, simplify_arguments=True))

        Dal1 = diff(loss(Y), Matrix([[al]]))
        Dtl1 = diff(loss(Y), Matrix([[tl]]))
        Dar1 = diff(loss(Y), Matrix([[ar]]))
        Dtr1 = diff(loss(Y), Matrix([[tr]]))

        # wrap values in a matrix to make them usable for the equal_matrices function
        Dal = Matrix([[Dal]])
        Dar = Matrix([[Dar]])
        Dtl = Matrix([[Dtl]])
        Dtr = Matrix([[Dtr]])
        self.assertTrue(equal_matrices(Dal, Dal1, simplify_arguments=True))
        self.assertTrue(equal_matrices(Dtl, Dtl1, simplify_arguments=True))
        self.assertTrue(equal_matrices(Dar, Dar1, simplify_arguments=False))
        self.assertTrue(equal_matrices(Dtr, Dtr1, simplify_arguments=True))


if __name__ == '__main__':
    import unittest
    unittest.main()
