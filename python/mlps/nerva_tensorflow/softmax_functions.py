# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import tensorflow as tf

from mlps.nerva_tensorflow.matrix_operations import Diag, column_repeat, columns_max, columns_sum, exp, hadamard, \
    identity, \
    reciprocal, is_column_vector, is_row_vector, log, row_repeat, rows_max, rows_sum

Matrix = tf.Tensor

def softmax_colwise(X: Matrix) -> Matrix:
    D, N = X.shape
    E = exp(X)
    return hadamard(E, row_repeat(reciprocal(columns_sum(E)), D))


def softmax_colwise_jacobian(x: Matrix) -> Matrix:
    assert is_column_vector(x)
    y = softmax_colwise(x)
    return Diag(y) - y * tf.transpose(y)


def stable_softmax_colwise(X: Matrix) -> Matrix:
    D, N = X.shape
    Y = X - row_repeat(columns_max(X), D)
    E = exp(Y)
    return hadamard(E, row_repeat(reciprocal(columns_sum(E)), D))


def stable_softmax_colwise_jacobian(x: Matrix) -> Matrix:
    assert is_column_vector(x)
    y = stable_softmax_colwise(x)
    return Diag(y) - y * y.T


def log_softmax_colwise(X: Matrix) -> Matrix:
    D, N = X.shape
    return X - row_repeat(log(columns_sum(exp(X))), D)


def stable_log_softmax_colwise(X: Matrix) -> Matrix:
    D, N = X.shape
    Y = X - row_repeat(columns_max(X), D)
    return Y - row_repeat(log(columns_sum(exp(Y))), D)


def log_softmax_colwise_jacobian(x: Matrix) -> Matrix:
    assert is_column_vector(x)
    D, N = x.shape
    return identity(D) - row_repeat(tf.transpose(softmax_colwise(x)), D)


def stable_log_softmax_colwise_jacobian(x: Matrix) -> Matrix:
    return log_softmax_colwise_jacobian(x)


def softmax_rowwise(X: Matrix) -> Matrix:
    N, D = X.shape
    E = exp(X)
    return hadamard(E, column_repeat(reciprocal(rows_sum(E)), D))


def softmax_rowwise_jacobian(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    y = softmax_rowwise(x)
    return Diag(y) - tf.transpose(y) * y


def stable_softmax_rowwise(X: Matrix) -> Matrix:
    N, D = X.shape
    Y = X - column_repeat(rows_max(X), D)
    E = exp(Y)
    return hadamard(E, column_repeat(reciprocal(rows_sum(E)), D))


def stable_softmax_rowwise_jacobian(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    y = stable_softmax_rowwise(x)
    return Diag(y) - y.T * y


def log_softmax_rowwise(X: Matrix) -> Matrix:
    N, D = X.shape
    return X - column_repeat(log(rows_sum(exp(X))), D)


def log_softmax_rowwise_jacobian(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    N, D = x.shape
    return identity(D) - row_repeat(softmax_rowwise(x), D)


def stable_log_softmax_rowwise(X: Matrix) -> Matrix:
    N, D = X.shape
    Y = X - column_repeat(rows_max(X), D)
    return Y - column_repeat(log(rows_sum(exp(Y))), D)


def stable_log_softmax_rowwise_jacobian(x: Matrix) -> Matrix:
    return log_softmax_rowwise_jacobian(x)
