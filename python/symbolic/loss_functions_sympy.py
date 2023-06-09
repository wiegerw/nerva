# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from symbolic.activation_functions_sympy import *
from symbolic.matrix_operations_sympy import *
from symbolic.softmax_sympy import *


def squared_error_loss(Y: Matrix, T: Matrix):
    return elements_sum(hadamard(Y - T, Y - T))


def squared_error_loss_gradient(Y: Matrix, T: Matrix):
    return 2 * (Y - T)


def mean_squared_error_loss(Y, T):
    return elements_sum(hadamard(Y - T, Y - T)) / len(Y)


def mean_squared_error_loss_gradient(Y, T):
    return 2 * (Y - T) / len(Y)


def cross_entropy_loss(Y, T):
    return -elements_sum(hadamard(T, log(Y)))


def cross_entropy_loss_gradient(Y, T):
    return -hadamard(T, inverse(Y))


def softmax_cross_entropy_loss(Y, T):
    return -elements_sum(hadamard(T, log_softmax_rowwise(Y)))


def softmax_cross_entropy_loss_gradient(Y, T):
    return softmax_rowwise(Y) - T


def stable_softmax_cross_entropy_loss(Y, T):
    return elements_sum(hadamard(-T, stable_log_softmax_rowwise(Y)))


def stable_softmax_cross_entropy_loss_gradient(Y, T):
    return stable_softmax_rowwise(Y) - T


def logistic_cross_entropy_loss(Y, T):
    return elements_sum(hadamard(-T, log(sigmoid(Y))))


def logistic_cross_entropy_loss_gradient(Y, T):
    N, K = Y.shape
    return hadamard(-T, ones(N, K) - sigmoid(Y))

