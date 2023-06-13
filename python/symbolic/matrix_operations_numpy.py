# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import numpy as np

def to_row(x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 1:
        return np.expand_dims(x, axis=0)
    return x


def to_col(x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 1:
        return np.expand_dims(x, axis=1)
    return x


def zeros(m: int, n: int = 1) -> np.ndarray:
    """
    Returns an mxn matrix with all elements equal to 0.
    """
    return np.zeros((m, n))


def ones(m: int, n: int = 1) -> np.ndarray:
    """
    Returns an mxn matrix with all elements equal to 1.
    """
    return np.ones((m, n))


def identity(n: int) -> np.ndarray:
    """
    Returns the nxn identity matrix.
    """
    return np.eye(n)


def product(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return X @ Y


def hadamard(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return X * Y


def diag(X: np.ndarray) -> np.ndarray:
    return to_col(np.diag(X))


def Diag(x: np.ndarray) -> np.ndarray:
    return np.diagflat(x)


def elements_sum(X: np.ndarray):
    """
    Returns the sum of the elements of X.
    """
    return np.sum(X)


def column_repeat(x: np.ndarray, n: int) -> np.ndarray:
    assert is_column_vector(x)
    return np.tile(x, (1, n))


def row_repeat(x: np.ndarray, m: int) -> np.ndarray:
    assert is_row_vector(x)
    return np.tile(x, (m, 1))


def columns_sum(X: np.ndarray) -> np.ndarray:
    return to_row(np.sum(X, axis=0))


def rows_sum(X: np.ndarray) -> np.ndarray:
    return to_col(np.sum(X, axis=1))


def columns_max(X: np.ndarray) -> np.ndarray:
    """
    Returns a column vector with the maximum values of each row in X.
    """
    return to_row(np.max(X, axis=0))


def rows_max(X: np.ndarray) -> np.ndarray:
    """
    Returns a row vector with the maximum values of each column in X.
    """
    return to_col(np.max(X, axis=1))


def columns_mean(X: np.ndarray) -> np.ndarray:
    """
    Returns a column vector with the mean values of each row in X.
    """
    return to_row(np.mean(X, axis=0))


def rows_mean(X: np.ndarray) -> np.ndarray:
    """
    Returns a row vector with the mean values of each column in X.
    """
    return to_col(np.mean(X, axis=1))


def apply(f, X: np.ndarray) -> np.ndarray:
    return f(X)


def exp(X: np.ndarray) -> np.ndarray:
    return np.exp(X)


def log(X: np.ndarray) -> np.ndarray:
    return np.log(X)


def inverse(X: np.ndarray) -> np.ndarray:
    return 1 / X


def square(X: np.ndarray) -> np.ndarray:
    return np.square(X)


def sqrt(X: np.ndarray) -> np.ndarray:
    return np.sqrt(X)


def power_minus_half(X: np.ndarray) -> np.ndarray:
    return inverse(sqrt(X))


###########################################################################################

def is_column_vector(x: np.ndarray) -> bool:
    m, n = x.shape
    return n == 1


def is_row_vector(x: np.ndarray) -> bool:
    m, n = x.shape
    return m == 1


def is_square(X: np.ndarray) -> bool:
    m, n = X.shape
    return m == n
