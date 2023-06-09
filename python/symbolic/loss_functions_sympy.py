# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from symbolic.matrix_operations_sympy import *


def squared_error_loss(Y: Matrix, T: Matrix):
    return elements_sum(hadamard(Y - T, Y - T))


def squared_error_loss_gradient(Y: Matrix, T: Matrix):
    return 2 * (Y - T)


def mean_squared_error_loss(Y, T):
    return elements_sum(hadamard(Y - T, Y - T)) / len(Y)


def mean_squared_error_loss_gradient(Y, T):
    return 2 * (Y - T) / len(Y)
