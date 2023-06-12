# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from symbolic.activation_functions_sympy import *
from symbolic.matrix_operations_sympy import *
from symbolic.softmax_sympy import *

def dot(x: Matrix, y: Matrix):
    assert is_column_vector(x) and is_column_vector(y)
    return (x.T * y)[0, 0]

#----------------------------------------------#
#         squared_error_loss_colwise
#----------------------------------------------#

def squared_error_loss_colwise_vector(y: Matrix, t: Matrix):
    return dot(y - t, y - t)


def squared_error_loss_colwise_gradient_vector(y: Matrix, t: Matrix):
    return 2 * (y - t)


def squared_error_loss(Y: Matrix, T: Matrix):
    return elements_sum(hadamard(Y - T, Y - T))


def squared_error_loss_colwise_gradient(Y: Matrix, T: Matrix):
    return 2 * (Y - T)

#----------------------------------------------#
#         mean_squared_error_loss_colwise
#----------------------------------------------#

def mean_squared_error_loss_colwise_vector(y: Matrix, t: Matrix):
    return dot(y - t, y - t) / len(y)


def mean_squared_error_loss_colwise_gradient_vector(y: Matrix, t: Matrix):
    return 2 * (y - t) / len(y)


def mean_squared_error_loss(Y, T):
    return elements_sum(hadamard(Y - T, Y - T)) / len(Y)


def mean_squared_error_loss_colwise_gradient(Y, T):
    return 2 * (Y - T) / len(Y)

#----------------------------------------------#
#         cross_entropy_loss_colwise
#----------------------------------------------#

def cross_entropy_loss_colwise_vector(y: Matrix, t: Matrix):
    return -dot(t, log(y))


def cross_entropy_loss_colwise_gradient_vector(y: Matrix, t: Matrix):
    return -hadamard(t, inverse(y))


def cross_entropy_loss(Y, T):
    return -elements_sum(hadamard(T, log(Y)))


def cross_entropy_loss_colwise_gradient(Y, T):
    return -hadamard(T, inverse(Y))

#----------------------------------------------#
#         softmax_cross_entropy_loss_colwise
#----------------------------------------------#

def softmax_cross_entropy_loss_colwise_vector(y: Matrix, t: Matrix):
    return -dot(t, log_softmax_colwise(y))


def softmax_cross_entropy_loss_colwise_gradient_vector(y: Matrix, t: Matrix):
    return elements_sum(t) * softmax_colwise(y) - t
    # K, N = y.shape
    # return column_repeat(softmax_colwise(y), K) * t - t
    # return -log_softmax_colwise_derivative(y).T * t


def softmax_cross_entropy_one_hot_loss_colwise_gradient_vector(y: Matrix, t: Matrix):
    """
    This is a simplified version of softmax_cross_entropy loss that requires t is a one-hot vector.
    """
    return softmax_colwise(y) - t


def softmax_cross_entropy_loss(Y, T):
    return -elements_sum(hadamard(T, log_softmax_colwise(Y)))


def softmax_cross_entropy_loss_colwise_gradient(Y, T):
    K, N = Y.shape
    return hadamard(softmax_colwise(Y), row_repeat(columns_sum(T), K)) - T


def softmax_cross_entropy_one_hot_loss_colwise_gradient(Y, T):
    return softmax_colwise(Y) - T

#----------------------------------------------#
#         stable_softmax_cross_entropy_loss_colwise
#----------------------------------------------#

def stable_softmax_cross_entropy_loss_colwise_vector(y: Matrix, t: Matrix):
    return -dot(t, stable_log_softmax_colwise(y))


def stable_softmax_cross_entropy_loss_colwise_gradient_vector(y: Matrix, t: Matrix):
    return elements_sum(t) * stable_softmax_colwise(y) - t
    # K, N = y.shape
    # return column_repeat(stable_softmax_colwise(y), K) * t - t
    # return -stable_log_softmax_colwise_derivative(y).T * t


def stable_softmax_cross_entropy_one_hot_loss_colwise_gradient_vector(y: Matrix, t: Matrix):
    """
    This is a simplified version of stable_softmax_cross_entropy loss that requires t is a one-hot vector.
    """
    return stable_softmax_colwise(y) - t


def stable_softmax_cross_entropy_loss(Y, T):
    return -elements_sum(hadamard(T, stable_log_softmax_colwise(Y)))


def stable_softmax_cross_entropy_loss_colwise_gradient(Y, T):
    K, N = Y.shape
    return hadamard(stable_softmax_colwise(Y), row_repeat(columns_sum(T), K)) - T


def stable_softmax_cross_entropy_one_hot_loss_colwise_gradient(Y, T):
    return stable_softmax_colwise(Y) - T

#----------------------------------------------#
#         logistic_cross_entropy_loss_colwise
#----------------------------------------------#

def logistic_cross_entropy_loss_colwise_vector(y: Matrix, t: Matrix):
    return -dot(t, log(sigmoid(y)))


def logistic_cross_entropy_loss_colwise_gradient_vector(y: Matrix, t: Matrix):
    return hadamard(t, sigmoid(y)) - t


def logistic_cross_entropy_loss(Y, T):
    return -elements_sum(hadamard(T, log(sigmoid(Y))))


def logistic_cross_entropy_loss_colwise_gradient(Y, T):
    return hadamard(T, sigmoid(Y)) - T

#----------------------------------------------#
#         negative_log_likelihood_loss_colwise
#----------------------------------------------#

def negative_log_likelihood_loss_colwise_vector(y: Matrix, t: Matrix):
    return -log(y.T * t)


def negative_log_likelihood_loss_colwise_gradient_vector(y: Matrix, t: Matrix):
    return (-1 / dot(y, t)) * t


def negative_log_likelihood_loss(Y, T):
    return -elements_sum(log(columns_sum(hadamard(Y, T))))


def negative_log_likelihood_loss_colwise_gradient(Y, T):
    K, N = Y.shape
    return -hadamard(row_repeat(inverse(columns_sum(hadamard(Y, T))), K), T)


