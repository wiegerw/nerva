# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from symbolic.activation_functions_sympy import *
from symbolic.softmax_sympy import *

def dot(x: Matrix, y: Matrix):
    if is_column_vector(x) and is_column_vector(y):
        return (x.T * y)[0, 0]
    elif is_row_vector(x) and is_row_vector(y):
        return (x * y.T)[0, 0]
    raise RuntimeError('dot: received illegal input')

#----------------------------------------------#
#         squared_error_loss_colwise
#----------------------------------------------#

def squared_error_loss_colwise_vector(y: Matrix, t: Matrix):
    return dot(y - t, y - t)


def squared_error_loss_colwise_gradient_vector(y: Matrix, t: Matrix):
    return 2 * (y - t)


def squared_error_loss_colwise(Y: Matrix, T: Matrix):
    return elements_sum(hadamard(Y - T, Y - T))


def squared_error_loss_colwise_gradient(Y: Matrix, T: Matrix):
    return 2 * (Y - T)

#----------------------------------------------#
#         squared_error_loss_rowwise
#----------------------------------------------#

def squared_error_loss_rowwise_vector(y: Matrix, t: Matrix):
    return dot(y - t, y - t)


def squared_error_loss_rowwise_gradient_vector(y: Matrix, t: Matrix):
    return 2 * (y - t)


def squared_error_loss_rowwise(Y: Matrix, T: Matrix):
    return elements_sum(hadamard(Y - T, Y - T))


def squared_error_loss_rowwise_gradient(Y: Matrix, T: Matrix):
    return 2 * (Y - T)

#----------------------------------------------#
#         mean_squared_error_loss_colwise
#----------------------------------------------#

def mean_squared_error_loss_colwise_vector(y: Matrix, t: Matrix):
    return dot(y - t, y - t) / len(y)


def mean_squared_error_loss_colwise_gradient_vector(y: Matrix, t: Matrix):
    return 2 * (y - t) / len(y)


def mean_squared_error_loss_colwise(Y: Matrix, T: Matrix):
    return elements_sum(hadamard(Y - T, Y - T)) / len(Y)


def mean_squared_error_loss_colwise_gradient(Y: Matrix, T: Matrix):
    return 2 * (Y - T) / len(Y)

#----------------------------------------------#
#         mean_squared_error_loss_rowwise
#----------------------------------------------#

def mean_squared_error_loss_rowwise_vector(y: Matrix, t: Matrix):
    return dot(y - t, y - t) / len(y)


def mean_squared_error_loss_rowwise_gradient_vector(y: Matrix, t: Matrix):
    return 2 * (y - t) / len(y)


def mean_squared_error_loss_rowwise(Y: Matrix, T: Matrix):
    return elements_sum(hadamard(Y - T, Y - T)) / len(Y)


def mean_squared_error_loss_rowwise_gradient(Y: Matrix, T: Matrix):
    return 2 * (Y - T) / len(Y)

#----------------------------------------------#
#         cross_entropy_loss_colwise
#----------------------------------------------#

def cross_entropy_loss_colwise_vector(y: Matrix, t: Matrix):
    return -dot(t, log(y))


def cross_entropy_loss_colwise_gradient_vector(y: Matrix, t: Matrix):
    return -hadamard(t, inverse(y))


def cross_entropy_loss_colwise(Y: Matrix, T: Matrix):
    return -elements_sum(hadamard(T, log(Y)))


def cross_entropy_loss_colwise_gradient(Y: Matrix, T: Matrix):
    return -hadamard(T, inverse(Y))

#----------------------------------------------#
#         cross_entropy_loss_rowwise
#----------------------------------------------#

def cross_entropy_loss_rowwise_vector(y: Matrix, t: Matrix):
    return -dot(t, log(y))


def cross_entropy_loss_rowwise_gradient_vector(y: Matrix, t: Matrix):
    return -hadamard(t, inverse(y))


def cross_entropy_loss_rowwise(Y: Matrix, T: Matrix):
    return -elements_sum(hadamard(T, log(Y)))


def cross_entropy_loss_rowwise_gradient(Y: Matrix, T: Matrix):
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


def softmax_cross_entropy_loss_colwise(Y: Matrix, T: Matrix):
    return -elements_sum(hadamard(T, log_softmax_colwise(Y)))


def softmax_cross_entropy_loss_colwise_gradient(Y: Matrix, T: Matrix):
    K, N = Y.shape
    return hadamard(softmax_colwise(Y), row_repeat(columns_sum(T), K)) - T


def softmax_cross_entropy_one_hot_loss_colwise_gradient(Y: Matrix, T: Matrix):
    return softmax_colwise(Y) - T

#----------------------------------------------#
#         softmax_cross_entropy_loss_rowwise
#----------------------------------------------#

def softmax_cross_entropy_loss_rowwise_vector(y: Matrix, t: Matrix):
    return -dot(t, log_softmax_rowwise(y))


def softmax_cross_entropy_loss_rowwise_gradient_vector(y: Matrix, t: Matrix):
    return softmax_rowwise(y) * elements_sum(t) - t


def softmax_cross_entropy_one_hot_loss_rowwise_gradient_vector(y: Matrix, t: Matrix):
    """
    This is a simplified version of softmax_cross_entropy loss that requires t is a one-hot vector.
    """
    return softmax_rowwise(y) - t


def softmax_cross_entropy_loss_rowwise(Y: Matrix, T: Matrix):
    return -elements_sum(hadamard(T, log_softmax_rowwise(Y)))


def softmax_cross_entropy_loss_rowwise_gradient(Y: Matrix, T: Matrix):
    N, K = Y.shape
    return hadamard(softmax_rowwise(Y), column_repeat(rows_sum(T), K)) - T


def softmax_cross_entropy_one_hot_loss_rowwise_gradient(Y: Matrix, T: Matrix):
    return softmax_rowwise(Y) - T

#----------------------------------------------#
#         stable_softmax_cross_entropy_loss_colwise
#----------------------------------------------#

def stable_softmax_cross_entropy_loss_colwise_vector(y: Matrix, t: Matrix):
    return -dot(t, stable_log_softmax_colwise(y))


def stable_softmax_cross_entropy_loss_colwise_gradient_vector(y: Matrix, t: Matrix):
    return elements_sum(t) * stable_softmax_colwise(y) - t


def stable_softmax_cross_entropy_one_hot_loss_colwise_gradient_vector(y: Matrix, t: Matrix):
    """
    This is a simplified version of stable_softmax_cross_entropy loss that requires t is a one-hot vector.
    """
    return stable_softmax_colwise(y) - t


def stable_softmax_cross_entropy_loss_colwise(Y: Matrix, T: Matrix):
    return -elements_sum(hadamard(T, stable_log_softmax_colwise(Y)))


def stable_softmax_cross_entropy_loss_colwise_gradient(Y: Matrix, T: Matrix):
    K, N = Y.shape
    return hadamard(stable_softmax_colwise(Y), row_repeat(columns_sum(T), K)) - T


def stable_softmax_cross_entropy_one_hot_loss_colwise_gradient(Y: Matrix, T: Matrix):
    return stable_softmax_colwise(Y) - T

#----------------------------------------------#
#         stable_softmax_cross_entropy_loss_rowwise
#----------------------------------------------#

def stable_softmax_cross_entropy_loss_rowwise_vector(y: Matrix, t: Matrix):
    return -dot(t, stable_log_softmax_rowwise(y))


def stable_softmax_cross_entropy_loss_rowwise_gradient_vector(y: Matrix, t: Matrix):
    return stable_softmax_rowwise(y) * elements_sum(t) - t


def stable_softmax_cross_entropy_one_hot_loss_rowwise_gradient_vector(y: Matrix, t: Matrix):
    """
    This is a simplified version of softmax_cross_entropy loss that requires t is a one-hot vector.
    """
    return stable_softmax_rowwise(y) - t


def stable_softmax_cross_entropy_loss_rowwise(Y: Matrix, T: Matrix):
    return -elements_sum(hadamard(T, stable_log_softmax_rowwise(Y)))


def stable_softmax_cross_entropy_loss_rowwise_gradient(Y: Matrix, T: Matrix):
    N, K = Y.shape
    return hadamard(stable_softmax_rowwise(Y), column_repeat(rows_sum(T), K)) - T


def stable_softmax_cross_entropy_one_hot_loss_rowwise_gradient(Y: Matrix, T: Matrix):
    return stable_softmax_rowwise(Y) - T


#----------------------------------------------#
#         logistic_cross_entropy_loss_colwise
#----------------------------------------------#

def logistic_cross_entropy_loss_colwise_vector(y: Matrix, t: Matrix):
    return -dot(t, log(sigmoid(y)))


def logistic_cross_entropy_loss_colwise_gradient_vector(y: Matrix, t: Matrix):
    return hadamard(t, sigmoid(y)) - t


def logistic_cross_entropy_loss_colwise(Y: Matrix, T: Matrix):
    return -elements_sum(hadamard(T, log(sigmoid(Y))))


def logistic_cross_entropy_loss_colwise_gradient(Y: Matrix, T: Matrix):
    return hadamard(T, sigmoid(Y)) - T

#----------------------------------------------#
#         logistic_cross_entropy_loss_rowwise
#----------------------------------------------#

def logistic_cross_entropy_loss_rowwise_vector(y: Matrix, t: Matrix):
    return -dot(t, log(sigmoid(y)))


def logistic_cross_entropy_loss_rowwise_gradient_vector(y: Matrix, t: Matrix):
    return hadamard(t, sigmoid(y)) - t


def logistic_cross_entropy_loss_rowwise(Y: Matrix, T: Matrix):
    return -elements_sum(hadamard(T, log(sigmoid(Y))))


def logistic_cross_entropy_loss_rowwise_gradient(Y: Matrix, T: Matrix):
    return hadamard(T, sigmoid(Y)) - T

#----------------------------------------------#
#         negative_log_likelihood_loss_colwise
#----------------------------------------------#

def negative_log_likelihood_loss_colwise_vector(y: Matrix, t: Matrix):
    return -sp.log(dot(y, t))


def negative_log_likelihood_loss_colwise_gradient_vector(y: Matrix, t: Matrix):
    return (-1 / dot(y, t)) * t


def negative_log_likelihood_loss_colwise(Y: Matrix, T: Matrix):
    return -elements_sum(log(columns_sum(hadamard(Y, T))))


def negative_log_likelihood_loss_colwise_gradient(Y, T):
    K, N = Y.shape
    return -hadamard(row_repeat(inverse(columns_sum(hadamard(Y, T))), K), T)

#----------------------------------------------#
#         negative_log_likelihood_loss_rowwise
#----------------------------------------------#

def negative_log_likelihood_loss_rowwise_vector(y: Matrix, t: Matrix):
    return -sp.log(dot(y, t))


def negative_log_likelihood_loss_rowwise_gradient_vector(y: Matrix, t: Matrix):
    return (-1 / dot(y, t)) * t


def negative_log_likelihood_loss_rowwise(Y: Matrix, T: Matrix):
    return -elements_sum(log(rows_sum(hadamard(Y, T))))


def negative_log_likelihood_loss_rowwise_gradient(Y, T):
    N, K = Y.shape
    return -hadamard(column_repeat(inverse(rows_sum(hadamard(Y, T))), K), T)
