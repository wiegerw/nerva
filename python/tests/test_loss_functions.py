#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase

import numpy as np

from symbolic.utilities import to_numpy, to_sympy, to_torch, to_tensorflow
import symbolic.loss_functions_numpy as np_
import symbolic.loss_functions_tensorflow as tf_
import symbolic.loss_functions_torch as torch_
import symbolic.loss_functions_sympy as sympy_

class TestLossFunctions(TestCase):
    def check_arrays_equal(self, operation, x1, x2, x3, x4):
        print(f'--- {operation} ---')
        x1 = to_numpy(x1)
        x2 = to_numpy(x2)
        x3 = to_numpy(x3)
        x4 = to_numpy(x4)
        print(x1)
        print(x2)
        print(x3)
        print(x4)
        self.assertTrue(np.allclose(x1, x2, atol=1e-5))
        self.assertTrue(np.allclose(x1, x3, atol=1e-5))
        self.assertTrue(np.allclose(x1, x4, atol=1e-5))

    def check_numbers_equal(self, operation, x1, x2, x3, x4):
        print(f'--- {operation} ---')
        print(x1, x1.__class__)
        print(x2, x2.__class__)
        print(x3, x3.__class__)
        print(x4, x4.__class__)
        self.assertTrue(x1 == x2)
        self.assertTrue(x1 == x3)
        self.assertTrue(x1 == x4)

    def make_variables(self):
        Y = np.array([
            [1, 2, 3],
            [7, 3, 4]
        ], dtype=float)

        T = np.array([
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=float)

        return Y, T

    def test_squared_error_loss(self):
        Y, T = self.make_variables()

        x1 = sympy_.squared_error_loss(to_sympy(Y), to_sympy(T))
        x2 = np_.squared_error_loss(to_numpy(Y), to_numpy(T))
        x3 = tf_.squared_error_loss(to_tensorflow(Y), to_tensorflow(T))
        x4 = torch_.squared_error_loss(to_torch(Y), to_torch(T))
        self.check_numbers_equal('squared_error_loss', x1, x2, x3, x4)

        x1 = sympy_.squared_error_loss_gradient(to_sympy(Y), to_sympy(T))
        x2 = np_.squared_error_loss_gradient(to_numpy(Y), to_numpy(T))
        x3 = tf_.squared_error_loss_gradient(to_tensorflow(Y), to_tensorflow(T))
        x4 = torch_.squared_error_loss_gradient(to_torch(Y), to_torch(T))
        self.check_arrays_equal('squared_error_loss_gradient', x1, x2, x3, x4)


if __name__ == '__main__':
    import unittest
    unittest.main()
