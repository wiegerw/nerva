# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import numpy as np

from mlps.nerva_numpy.activation_functions import Sigmoid
from mlps.nerva_numpy.matrix_operations import column_repeat, columns_sum, dot, elements_sum, hadamard, inverse, log, \
    row_repeat, rows_sum
from mlps.nerva_numpy.softmax_functions import log_softmax_colwise, log_softmax_rowwise, softmax_colwise, \
    softmax_rowwise, \
    stable_log_softmax_colwise, stable_log_softmax_rowwise, stable_softmax_colwise, stable_softmax_rowwise


# Naming conventions:
# - lowercase functions operate on vectors (y and t)
# - uppercase functions operate on matrices (Y and T)


def squared_error_loss_colwise(y, t):
    return dot(y - t, y - t)


def squared_error_loss_colwise_gradient(y, t):
    return 2 * (y - t)


def Squared_error_loss_colwise(Y, T):
    return elements_sum(hadamard(Y - T, Y - T))


def Squared_error_loss_colwise_gradient(Y, T):
    return 2 * (Y - T)


def mean_squared_error_loss_colwise(y, t):
    K, N = y.shape
    return squared_error_loss_colwise(y, t) / K


def mean_squared_error_loss_colwise_gradient(y, t):
    K, N = y.shape
    return squared_error_loss_colwise_gradient(y, t) / K


def Mean_squared_error_loss_colwise(Y, T):
    K, N = Y.shape
    return Squared_error_loss_colwise(Y, T) / (K * N)


def Mean_squared_error_loss_colwise_gradient(Y, T):
    K, N = Y.shape
    return Squared_error_loss_colwise_gradient(Y, T) / (K * N)


def cross_entropy_loss_colwise(y, t):
    return -dot(t, log(y))


def cross_entropy_loss_colwise_gradient(y, t):
    return -hadamard(t, inverse(y))


def Cross_entropy_loss_colwise(Y, T):
    return -elements_sum(hadamard(T, log(Y)))


def Cross_entropy_loss_colwise_gradient(Y, T):
    return -hadamard(T, inverse(Y))


def softmax_cross_entropy_loss_colwise(y, t):
    return -dot(t, log_softmax_colwise(y))


def softmax_cross_entropy_loss_colwise_gradient(y, t):
    return elements_sum(t) * softmax_colwise(y) - t


def softmax_cross_entropy_loss_colwise_gradient_one_hot(y, t):
    return softmax_colwise(y) - t


def Softmax_cross_entropy_loss_colwise(Y, T):
    return -elements_sum(hadamard(T, log_softmax_colwise(Y)))


def Softmax_cross_entropy_loss_colwise_gradient(Y, T):
    K, N = Y.shape
    return hadamard(softmax_colwise(Y), row_repeat(columns_sum(T), K)) - T


def Softmax_cross_entropy_loss_colwise_gradient_one_hot(Y, T):
    return softmax_colwise(Y) - T


def stable_softmax_cross_entropy_loss_colwise(y, t):
    return -dot(t, stable_log_softmax_colwise(y))


def stable_softmax_cross_entropy_loss_colwise_gradient(y, t):
    return elements_sum(t) * stable_softmax_colwise(y) - t


def stable_softmax_cross_entropy_loss_colwise_gradient_one_hot(y, t):
    return stable_softmax_colwise(y) - t


def Stable_softmax_cross_entropy_loss_colwise(Y, T):
    return -elements_sum(hadamard(T, stable_log_softmax_colwise(Y)))


def Stable_softmax_cross_entropy_loss_colwise_gradient(Y, T):
    K, N = Y.shape
    return hadamard(stable_softmax_colwise(Y), row_repeat(columns_sum(T), K)) - T


def Stable_softmax_cross_entropy_loss_colwise_gradient_one_hot(Y, T):
    return stable_softmax_colwise(Y) - T


def logistic_cross_entropy_loss_colwise(y, t):
    return -dot(t, log(Sigmoid(y)))


def logistic_cross_entropy_loss_colwise_gradient(y, t):
    return hadamard(t, Sigmoid(y)) - t


def Logistic_cross_entropy_loss_colwise(Y, T):
    return -elements_sum(hadamard(T, log(Sigmoid(Y))))


def Logistic_cross_entropy_loss_colwise_gradient(Y, T):
    return hadamard(T, Sigmoid(Y)) - T


def negative_log_likelihood_loss_colwise(y, t):
    return -np.log(dot(y, t))


def negative_log_likelihood_loss_colwise_gradient(y, t):
    return (-1 / dot(y, t)) * t


def Negative_log_likelihood_loss_colwise(Y, T):
    return -elements_sum(log(columns_sum(hadamard(Y, T))))


def Negative_log_likelihood_loss_colwise_gradient(Y, T):
    K, N = Y.shape
    return -hadamard(row_repeat(inverse(columns_sum(hadamard(Y, T))), K), T)


def squared_error_loss_rowwise(y, t):
    return dot(y - t, y - t)


def squared_error_loss_rowwise_gradient(y, t):
    return 2 * (y - t)


def Squared_error_loss_rowwise(Y, T):
    return elements_sum(hadamard(Y - T, Y - T))


def Squared_error_loss_rowwise_gradient(Y, T):
    return 2 * (Y - T)


def mean_squared_error_loss_rowwise(y, t):
    N, K = y.shape
    return squared_error_loss_rowwise(y, t) / K


def mean_squared_error_loss_rowwise_gradient(y, t):
    N, K = y.shape
    return squared_error_loss_rowwise_gradient(y, t) / K


def Mean_squared_error_loss_rowwise(Y, T):
    N, K = Y.shape
    return Squared_error_loss_rowwise(Y, T) / (K * N)


def Mean_squared_error_loss_rowwise_gradient(Y, T):
    N, K = Y.shape
    return Squared_error_loss_rowwise_gradient(Y, T) / (K * N)


def cross_entropy_loss_rowwise(y, t):
    return -dot(t, log(y))


def cross_entropy_loss_rowwise_gradient(y, t):
    return -hadamard(t, inverse(y))


def Cross_entropy_loss_rowwise(Y, T):
    return -elements_sum(hadamard(T, log(Y)))


def Cross_entropy_loss_rowwise_gradient(Y, T):
    return -hadamard(T, inverse(Y))


def softmax_cross_entropy_loss_rowwise(y, t):
    return -dot(t, log_softmax_rowwise(y))


def softmax_cross_entropy_loss_rowwise_gradient(y, t):
    return softmax_rowwise(y) * elements_sum(t) - t


def softmax_cross_entropy_loss_rowwise_gradient_one_hot(y, t):
    return softmax_rowwise(y) - t


def Softmax_cross_entropy_loss_rowwise(Y, T):
    return -elements_sum(hadamard(T, log_softmax_rowwise(Y)))


def Softmax_cross_entropy_loss_rowwise_gradient(Y, T):
    N, K = Y.shape
    return hadamard(softmax_rowwise(Y), column_repeat(rows_sum(T), K)) - T

def Softmax_cross_entropy_loss_rowwise_gradient_one_hot(Y, T):
    return softmax_rowwise(Y) - T


def stable_softmax_cross_entropy_loss_rowwise(y, t):
    return -dot(t, stable_log_softmax_rowwise(y))


def stable_softmax_cross_entropy_loss_rowwise_gradient(y, t):
    return stable_softmax_rowwise(y) * elements_sum(t) - t


def stable_softmax_cross_entropy_loss_rowwise_gradient_one_hot(y, t):
    return stable_softmax_rowwise(y) - t


def Stable_softmax_cross_entropy_loss_rowwise(Y, T):
    return -elements_sum(hadamard(T, stable_log_softmax_rowwise(Y)))


def Stable_softmax_cross_entropy_loss_rowwise_gradient(Y, T):
    N, K = Y.shape
    return hadamard(stable_softmax_rowwise(Y), column_repeat(rows_sum(T), K)) - T


def Stable_softmax_cross_entropy_loss_rowwise_gradient_one_hot(Y, T):
    return stable_softmax_rowwise(Y) - T


def logistic_cross_entropy_loss_rowwise(y, t):
    return -dot(t, log(Sigmoid(y)))


def logistic_cross_entropy_loss_rowwise_gradient(y, t):
    return hadamard(t, Sigmoid(y)) - t


def Logistic_cross_entropy_loss_rowwise(Y, T):
    return -elements_sum(hadamard(T, log(Sigmoid(Y))))


def Logistic_cross_entropy_loss_rowwise_gradient(Y, T):
    return hadamard(T, Sigmoid(Y)) - T


def negative_log_likelihood_loss_rowwise(y, t):
    return -np.log(dot(y, t))


def negative_log_likelihood_loss_rowwise_gradient(y, t):
    return (-1 / dot(y, t)) * t


def Negative_log_likelihood_loss_rowwise(Y, T):
    return -elements_sum(log(rows_sum(hadamard(Y, T))))


def Negative_log_likelihood_loss_rowwise_gradient(Y, T):
    N, K = Y.shape
    return -hadamard(column_repeat(inverse(rows_sum(hadamard(Y, T))), K), T)
