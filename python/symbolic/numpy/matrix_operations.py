# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import numpy as np

Matrix = np.ndarray

def is_vector(x: Matrix) -> bool:
    return len(x.shape) == 1


def is_square(X: Matrix) -> bool:
    m, n = X.shape
    return m == n


def dot(x: Matrix, y: Matrix):
    return x.T @ y


def zeros(m: int, n=None) -> Matrix:
    """
    Returns an mxn matrix with all elements equal to 0.
    """
    return np.zeros((m, n)) if n else np.zeros(m)


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
    return np.diag(X)


def Diag(x: Matrix) -> Matrix:
    return np.diagflat(x)


def elements_sum(X: Matrix):
    """
    Returns the sum of the elements of X.
    """
    return np.sum(X)


def column_repeat(x: Matrix, n: int) -> Matrix:
    assert is_vector(x)
    return np.tile(x[:, np.newaxis], (1, n))


def row_repeat(x: Matrix, m: int) -> Matrix:
    assert is_vector(x)
    return np.tile(x[np.newaxis, :], (m, 1))


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


def inverse(X: Matrix) -> Matrix:
    return 1 / X


def square(X: Matrix) -> Matrix:
    return np.square(X)


def sqrt(X: Matrix) -> Matrix:
    return np.sqrt(X)


def power_minus_half(X: Matrix) -> Matrix:
    return inverse(sqrt(X))
