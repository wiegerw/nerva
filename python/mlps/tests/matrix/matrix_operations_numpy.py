# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import numpy as np

Matrix = np.ndarray

def is_column_vector(x: Matrix) -> bool:
    m, n = x.shape
    return n == 1


def is_row_vector(x: Matrix) -> bool:
    m, n = x.shape
    return m == 1


def is_square(X: Matrix) -> bool:
    m, n = X.shape
    return m == n


def dot(x: Matrix, y: Matrix):
    if is_column_vector(x) and is_column_vector(y):
        return x.T @ y
    elif is_row_vector(x) and is_row_vector(y):
        return x @ y.T
    raise RuntimeError('dot: received illegal input')


def to_row(x: Matrix) -> Matrix:
    if len(x.shape) == 1:
        return np.expand_dims(x, axis=0)
    return x


def to_col(x: Matrix) -> Matrix:
    if len(x.shape) == 1:
        return np.expand_dims(x, axis=1)
    return x


def zeros(m: int, n: int = 1) -> Matrix:
    """
    Returns an mxn matrix with all elements equal to 0.
    """
    return np.zeros((m, n))


def ones(m: int, n: int = 1) -> Matrix:
    """
    Returns an mxn matrix with all elements equal to 1.
    """
    return np.ones((m, n))


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
    return to_col(np.diag(X))


def Diag(x: Matrix) -> Matrix:
    return np.diagflat(x)


def elements_sum(X: Matrix):
    """
    Returns the sum of the elements of X.
    """
    return np.sum(X)


def column_repeat(x: Matrix, n: int) -> Matrix:
    assert is_column_vector(x)
    return np.tile(x, (1, n))


def row_repeat(x: Matrix, m: int) -> Matrix:
    assert is_row_vector(x)
    return np.tile(x, (m, 1))


def columns_sum(X: Matrix) -> Matrix:
    return to_row(np.sum(X, axis=0))


def rows_sum(X: Matrix) -> Matrix:
    return to_col(np.sum(X, axis=1))


def columns_max(X: Matrix) -> Matrix:
    """
    Returns a column vector with the maximum values of each row in X.
    """
    return to_row(np.max(X, axis=0))


def rows_max(X: Matrix) -> Matrix:
    """
    Returns a row vector with the maximum values of each column in X.
    """
    return to_col(np.max(X, axis=1))


def columns_mean(X: Matrix) -> Matrix:
    """
    Returns a column vector with the mean values of each row in X.
    """
    return to_row(np.mean(X, axis=0))


def rows_mean(X: Matrix) -> Matrix:
    """
    Returns a row vector with the mean values of each column in X.
    """
    return to_col(np.mean(X, axis=1))


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


def div_sqrt(X: Matrix) -> Matrix:
    return reciprocal(sqrt(X))
