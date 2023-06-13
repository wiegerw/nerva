# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from symbolic.matrix_operations_tensorflow import *

def softmax_colwise(X: tf.Tensor) -> tf.Tensor:
    D, N = X.shape
    E = exp(X)
    return hadamard(E, row_repeat(inverse(columns_sum(E)), D))


def softmax_colwise_jacobian(x: tf.Tensor) -> tf.Tensor:
    assert is_column_vector(x)
    y = softmax_colwise(x)
    return Diag(y) - y * tf.transpose(y)  # TODO: y.T does not work here!


def stable_softmax_colwise(X: tf.Tensor) -> tf.Tensor:
    D, N = X.shape
    Y = X - row_repeat(columns_max(X), D)
    E = exp(Y)
    return hadamard(E, row_repeat(inverse(columns_sum(E)), D))


def log_softmax_colwise(X: tf.Tensor) -> tf.Tensor:
    D, N = X.shape
    return X - row_repeat(log(columns_sum(exp(X))), D)


def stable_log_softmax_colwise(X: tf.Tensor) -> tf.Tensor:
    D, N = X.shape
    Y = X - row_repeat(columns_max(X), D)
    return Y - row_repeat(log(columns_sum(exp(Y))), D)


def log_softmax_colwise_jacobian(x: tf.Tensor) -> tf.Tensor:
    assert is_column_vector(x)
    D, N = x.shape
    return identity(D) - row_repeat(tf.transpose(softmax_colwise(x)), D)


def softmax_rowwise(X: tf.Tensor) -> tf.Tensor:
    N, D = X.shape
    E = exp(X)
    return hadamard(E, column_repeat(inverse(rows_sum(E)), D))


def softmax_rowwise_jacobian(x: tf.Tensor) -> tf.Tensor:
    assert is_row_vector(x)
    y = softmax_rowwise(x)
    return Diag(y) - tf.transpose(y) * y


def stable_softmax_rowwise(X: tf.Tensor) -> tf.Tensor:
    N, D = X.shape
    Y = X - column_repeat(rows_max(X), D)
    E = exp(Y)
    return hadamard(E, column_repeat(inverse(rows_sum(E)), D))


def log_softmax_rowwise(X: tf.Tensor) -> tf.Tensor:
    N, D = X.shape
    return X - column_repeat(log(rows_sum(exp(X))), D)


def log_softmax_rowwise_jacobian(x: tf.Tensor) -> tf.Tensor:
    assert is_row_vector(x)
    N, D = x.shape
    return identity(D) - row_repeat(softmax_rowwise(x), D)


def stable_log_softmax_rowwise(X: tf.Tensor) -> tf.Tensor:
    N, D = X.shape
    Y = X - column_repeat(rows_max(X), D)
    return Y - column_repeat(log(rows_sum(exp(Y))), D)

