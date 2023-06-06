# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from symbolic.matrix_operations import *

#-------------------------------------#
#           softmax colwise
#-------------------------------------#

def softmax_colwise(X: Matrix) -> Matrix:
    D, N = X.shape
    E = exp(X)
    return hadamard(E, repeat_row(inverse(sum_columns(E)), D))


def stable_softmax_colwise(X: Matrix) -> Matrix:
    D, N = X.shape
    Y = X - repeat_row(column_max_values(X), D)
    E = exp(Y)
    return hadamard(E, repeat_row(inverse(sum_columns(E)), D))


def softmax_colwise_derivative(x: Matrix) -> Matrix:
    assert is_column_vector(x)
    y = softmax_colwise(x)
    return Diag(y) - y * y.T


#-------------------------------------#
#           log_softmax colwise
#-------------------------------------#

def log_softmax_colwise(X: Matrix) -> Matrix:
    D, N = X.shape
    return X - repeat_row(log(sum_columns(exp(X))), D)


def stable_log_softmax_colwise(X: Matrix) -> Matrix:
    D, N = X.shape
    Y = X - repeat_row(column_max_values(X), D)
    return Y - repeat_row(log(sum_columns(exp(Y))), D)


def log_softmax_colwise_derivative(x: Matrix) -> Matrix:
    assert is_column_vector(x)
    D, N = x.shape
    return identity(D) - repeat_row(softmax_colwise(x).T, D)


#-------------------------------------#
#           softmax rowwise
#-------------------------------------#

def softmax_rowwise(X: Matrix) -> Matrix:
    N, D = X.shape
    E = exp(X)
    return hadamard(E, repeat_column(inverse(sum_rows(E)), D))


def stable_softmax_rowwise(X: Matrix) -> Matrix:
    N, D = X.shape
    Y = X - repeat_column(row_max_values(X), D)
    E = exp(Y)
    return hadamard(E, repeat_column(inverse(sum_rows(E)), D))


def softmax_rowwise_derivative(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    y = softmax_rowwise(x)
    return Diag(y) - y.T * y


#-------------------------------------#
#           log_softmax rowwise
#-------------------------------------#

def log_softmax_rowwise(X: Matrix) -> Matrix:
    N, D = X.shape
    return X - repeat_column(log(sum_rows(exp(X))), D)


def stable_log_softmax_rowwise(X: Matrix) -> Matrix:
    N, D = X.shape
    Y = X - repeat_column(row_max_values(X), D)
    return Y - repeat_column(log(sum_rows(exp(Y))), D)


def log_softmax_rowwise_derivative(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    N, D = x.shape
    return identity(D) - repeat_row(softmax_rowwise(x), D)
