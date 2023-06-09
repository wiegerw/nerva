# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from matrix_operations_numpy import *


def softmax_colwise(X: np.ndarray) -> np.ndarray:
    D, N = X.shape
    E = exp(X)
    return hadamard(E, row_repeat(inverse(columns_sum(E)), D))


def stable_softmax_colwise(X: np.ndarray) -> np.ndarray:
    D, N = X.shape
    Y = X - row_repeat(columns_max(X), D)
    E = exp(Y)
    return hadamard(E, row_repeat(inverse(columns_sum(E)), D))


def log_softmax_colwise(X: np.ndarray) -> np.ndarray:
    D, N = X.shape
    return X - row_repeat(log(columns_sum(exp(X))), D)


def stable_log_softmax_colwise(X: np.ndarray) -> np.ndarray:
    D, N = X.shape
    Y = X - row_repeat(columns_max(X), D)
    return Y - row_repeat(log(columns_sum(exp(Y))), D)


def log_softmax_colwise_derivative(x: np.ndarray) -> np.ndarray:
    assert is_column_vector(x)
    D, N = x.shape
    return identity(D) - row_repeat(softmax_colwise(x).T, D)


def softmax_rowwise(X: np.ndarray) -> np.ndarray:
    N, D = X.shape
    E = exp(X)
    return hadamard(E, column_repeat(inverse(rows_sum(E)), D))


def softmax_rowwise1(X: np.ndarray) -> np.ndarray:
    return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)


def stable_softmax_rowwise(X: np.ndarray) -> np.ndarray:
    N, D = X.shape
    Y = X - column_repeat(rows_max(X), D)
    E = exp(Y)
    return hadamard(E, column_repeat(inverse(rows_sum(E)), D))


def stable_softmax_rowwise1(X: np.ndarray) -> np.ndarray:
    Y = X - np.max(X, axis=1, keepdims=True)
    return np.exp(Y) / np.sum(np.exp(Y), axis=1, keepdims=True)


def log_softmax_rowwise(X: np.ndarray) -> np.ndarray:
    N, D = X.shape
    return X - column_repeat(log(rows_sum(exp(X))), D)


def stable_log_softmax_rowwise(X: np.ndarray) -> np.ndarray:
    N, D = X.shape
    Y = X - column_repeat(rows_max(X), D)
    return Y - column_repeat(log(rows_sum(exp(Y))), D)


def log_softmax_rowwise_derivative(x: np.ndarray) -> np.ndarray:
    assert is_row_vector(x)
    N, D = x.shape
    return identity(D) - row_repeat(softmax_rowwise(x), D)
