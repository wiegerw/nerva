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


class SquaredErrorLossColwise(object):

    def vector_value(self, y, t):
        return dot(y - t, y - t)

    def vector_gradient(self, y, t):
        return 2 * (y - t)

    def value(self, Y, T):
        return elements_sum(hadamard(Y - T, Y - T))

    def gradient(self, Y, T):
        return 2 * (Y - T)


class MeanSquaredErrorLossColwise(object):

    def vector_value(self, y, t):
        K, N = y.shape
        return SquaredErrorLossColwise().vector_value(y, t) / K

    def vector_gradient(self, y, t):
        K, N = y.shape
        return SquaredErrorLossColwise().vector_gradient(y, t) / K

    def value(self, Y, T):
        K, N = Y.shape
        return SquaredErrorLossColwise().value(Y, T) / (K * N)

    def gradient(self, Y, T):
        K, N = Y.shape
        return SquaredErrorLossColwise().gradient(Y, T) / (K * N)


class CrossEntropyLossColwise(object):

    def vector_value(self, y, t):
        return -dot(t, log(y))

    def vector_gradient(self, y, t):
        return -hadamard(t, inverse(y))

    def value(self, Y, T):
        return -elements_sum(hadamard(T, log(Y)))

    def gradient(self, Y, T):
        return -hadamard(T, inverse(Y))


class SoftmaxCrossEntropyLossColwise(object):

    def vector_value(self, y, t):
        return -dot(t, log_softmax_colwise(y))

    def vector_gradient(self, y, t):
        return elements_sum(t) * softmax_colwise(y) - t

    def vector_gradient_one_hot(self, y, t):
        return softmax_colwise(y) - t

    def value(self, Y, T):
        return -elements_sum(hadamard(T, log_softmax_colwise(Y)))

    def gradient(self, Y, T):
        K, N = Y.shape
        return hadamard(softmax_colwise(Y), row_repeat(columns_sum(T), K)) - T

    def gradient_one_hot(self, Y, T):
        return softmax_colwise(Y) - T


class StableSoftmaxCrossEntropyLossColwise(object):

    def vector_value(self, y, t):
        return -dot(t, stable_log_softmax_colwise(y))

    def vector_gradient(self, y, t):
        return elements_sum(t) * stable_softmax_colwise(y) - t

    def vector_gradient_one_hot(self, y, t):
        return stable_softmax_colwise(y) - t

    def value(self, Y, T):
        return -elements_sum(hadamard(T, stable_log_softmax_colwise(Y)))

    def gradient(self, Y, T):
        K, N = Y.shape
        return hadamard(stable_softmax_colwise(Y), row_repeat(columns_sum(T), K)) - T

    def gradient_one_hot(self, Y, T):
        return stable_softmax_colwise(Y) - T


class LogisticCrossEntropyLossColwise(object):

    def vector_value(self, y, t):
        return -dot(t, log(Sigmoid(y)))

    def vector_gradient(self, y, t):
        return hadamard(t, Sigmoid(y)) - t

    def value(self, Y, T):
        return -elements_sum(hadamard(T, log(Sigmoid(Y))))

    def gradient(self, Y, T):
        return hadamard(T, Sigmoid(Y)) - T


class NegativeLogLikelihoodLossColwise(object):

    def vector_value(self, y, t):
        return -sp.log(dot(y, t))

    def vector_gradient(self, y, t):
        return (-1 / dot(y, t)) * t

    def value(self, Y, T):
        return -elements_sum(log(columns_sum(hadamard(Y, T))))

    def gradient(self, Y, T):
        K, N = Y.shape
        return -hadamard(row_repeat(inverse(columns_sum(hadamard(Y, T))), K), T)

#####################################################################################


class SquaredErrorLossRowwise(object):

    def vector_value(self, y, t):
        return dot(y - t, y - t)

    def vector_gradient(self, y, t):
        return 2 * (y - t)

    def value(self, Y, T):
        return elements_sum(hadamard(Y - T, Y - T))

    def gradient(self, Y, T):
        return 2 * (Y - T)


class MeanSquaredErrorLossRowwise(object):

    def vector_value(self, y, t):
        N, K = y.shape
        return SquaredErrorLossRowwise().vector_value(y, t) / K

    def vector_gradient(self, y, t):
        N, K = y.shape
        return SquaredErrorLossRowwise().vector_gradient(y, t) / K

    def value(self, Y, T):
        N, K = Y.shape
        return SquaredErrorLossRowwise().value(Y, T) / (K * N)

    def gradient(self, Y, T):
        N, K = Y.shape
        return SquaredErrorLossRowwise().gradient(Y, T) / (K * N)


class CrossEntropyLossRowwise(object):

    def vector_value(self, y, t):
        return -dot(t, log(y))

    def vector_gradient(self, y, t):
        return -hadamard(t, inverse(y))

    def value(self, Y, T):
        return -elements_sum(hadamard(T, log(Y)))

    def gradient(self, Y, T):
        return -hadamard(T, inverse(Y))


class SoftmaxCrossEntropyLossRowwise(object):

    def vector_value(self, y, t):
        return -dot(t, log_softmax_rowwise(y))

    def vector_gradient(self, y, t):
        return softmax_rowwise(y) * elements_sum(t) - t

    def vector_gradient_one_hot(self, y, t):
        return softmax_rowwise(y) - t

    def value(self, Y, T):
        return -elements_sum(hadamard(T, log_softmax_rowwise(Y)))

    def gradient(self, Y, T):
        N, K = Y.shape
        return hadamard(softmax_rowwise(Y), column_repeat(rows_sum(T), K)) - T

    def gradient_one_hot(self, Y, T):
        return softmax_rowwise(Y) - T


class StableSoftmaxCrossEntropyLossRowwise(object):

    def vector_value(self, y, t):
        return -dot(t, stable_log_softmax_rowwise(y))

    def vector_gradient(self, y, t):
        return stable_softmax_rowwise(y) * elements_sum(t) - t

    def vector_gradient_one_hot(self, y, t):
        return stable_softmax_rowwise(y) - t

    def value(self, Y, T):
        return -elements_sum(hadamard(T, stable_log_softmax_rowwise(Y)))

    def gradient(self, Y, T):
        N, K = Y.shape
        return hadamard(stable_softmax_rowwise(Y), column_repeat(rows_sum(T), K)) - T

    def gradient_one_hot(self, Y, T):
        return stable_softmax_rowwise(Y) - T


class LogisticCrossEntropyLossRowwise(object):

    def vector_value(self, y, t):
        return -dot(t, log(Sigmoid(y)))

    def vector_gradient(self, y, t):
        return hadamard(t, Sigmoid(y)) - t

    def value(self, Y, T):
        return -elements_sum(hadamard(T, log(Sigmoid(Y))))

    def gradient(self, Y, T):
        return hadamard(T, Sigmoid(Y)) - T


class NegativeLogLikelihoodLossRowwise(object):

    def vector_value(self, y, t):
        return -sp.log(dot(y, t))

    def vector_gradient(self, y, t):
        return (-1 / dot(y, t)) * t

    def value(self, Y, T):
        return -elements_sum(log(rows_sum(hadamard(Y, T))))

    def gradient(self, Y, T):
        N, K = Y.shape
        return -hadamard(column_repeat(inverse(rows_sum(hadamard(Y, T))), K), T)
