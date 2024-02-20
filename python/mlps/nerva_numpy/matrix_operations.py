# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import numpy as np

Matrix = np.ndarray


# A constant used by inv_sqrt to avoid division by zero
epsilon = 1e-7


def is_vector(x: Matrix) -> bool:
    return len(x.shape) == 1


def is_column_vector(x: Matrix) -> bool:
    return is_vector(x) or x.shape[1] == 1


def is_row_vector(x: Matrix) -> bool:
    return is_vector(x) or x.shape[0] == 1


def vector_size(x: Matrix) -> int:
    return x.shape[0]


def is_square(X: Matrix) -> bool:
    m, n = X.shape
    return m == n


def dot(x: Matrix, y: Matrix):
    return np.dot(np.squeeze(x), np.squeeze(y))


def zeros(m: int, n=None) -> Matrix:
    """
    Returns an mxn matrix with all elements equal to 0.
    """
    return np.zeros((m, n)) if n else np.zeros(m)


def ones(m: int, n=None) -> Matrix:
    """
    Returns an mxn matrix with all elements equal to 1.
    """
    return np.ones((m, n)) if n else np.ones(m)


def identity(n: int) -> Matrix:
    """
    Returns the nxn identity matrix.
    """
    return np.eye(n)


def product(X: Matrix, Y: Matrix) -> Matrix:
    return X @ Y


def hadamard(X: Matrix, Y: Matrix) -> Matrix:
    return X * Y


def diag(X: Matrix) -> Matrix:
    return np.diag(X)


def Diag(x: Matrix) -> Matrix:
    return np.diagflat(x)


def elements_sum(X: Matrix):
    """
    Returns the sum of the elements of X.
    """
    return np.sum(X)


def column_repeat(x: Matrix, n: int) -> Matrix:
    assert is_column_vector(x)
    if len(x.shape) == 1:
        x = x[:, np.newaxis]
    return np.tile(x, (1, n))


def row_repeat(x: Matrix, m: int) -> Matrix:
    assert is_row_vector(x)
    if len(x.shape) == 1:
        x = x[np.newaxis, :]
    return np.tile(x, (m, 1))


def columns_sum(X: Matrix) -> Matrix:
    return np.sum(X, axis=0)


def rows_sum(X: Matrix) -> Matrix:
    return np.sum(X, axis=1)


def columns_max(X: Matrix) -> Matrix:
    """
    Returns a column vector with the maximum values of each row in X.
    """
    return np.max(X, axis=0)


def rows_max(X: Matrix) -> Matrix:
    """
    Returns a row vector with the maximum values of each column in X.
    """
    return np.max(X, axis=1)


def columns_mean(X: Matrix) -> Matrix:
    """
    Returns a column vector with the mean values of each row in X.
    """
    return np.mean(X, axis=0)


def rows_mean(X: Matrix) -> Matrix:
    """
    Returns a row vector with the mean values of each column in X.
    """
    return np.mean(X, axis=1)


def apply(f, X: Matrix) -> Matrix:
    return f(X)


def exp(X: Matrix) -> Matrix:
    return np.exp(X)


def log(X: Matrix) -> Matrix:
    return np.log(X)


def reciprocal(X: Matrix) -> Matrix:
    return 1 / X


def square(X: Matrix) -> Matrix:
    return np.square(X)


def sqrt(X: Matrix) -> Matrix:
    return np.sqrt(X)


def inv_sqrt(X: Matrix) -> Matrix:
    return 1 / np.sqrt(X + epsilon)  # The epsilon is needed for numerical stability


def log_sigmoid(X: Matrix) -> Matrix:
    return -np.logaddexp(0, -X)
