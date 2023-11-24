# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import torch

def is_column_vector(x: torch.Tensor) -> bool:
    m, n = x.shape
    return n == 1


def is_row_vector(x: torch.Tensor) -> bool:
    m, n = x.shape
    return m == 1


def is_square(X: torch.Tensor) -> bool:
    m, n = X.shape
    return m == n


def dot(x, y):
    if is_column_vector(x) and is_column_vector(y):
        return x.T @ y
    elif is_row_vector(x) and is_row_vector(y):
        return x @ y.T
    raise RuntimeError('dot: received illegal input')


def to_row(x: torch.Tensor) -> torch.Tensor:
    if len(x.shape) == 1:
        return x.unsqueeze(dim=0)
    return x


def to_col(x: torch.Tensor) -> torch.Tensor:
    if len(x.shape) == 1:
        return x.unsqueeze(dim=1)
    return x


def zeros(m: int, n: int = 1) -> torch.Tensor:
    """
    Returns an mxn matrix with all elements equal to 0.
    """
    return torch.zeros(m, n)


def ones(m: int, n: int = 1) -> torch.Tensor:
    """
    Returns an mxn matrix with all elements equal to 1.
    """
    return torch.ones(m, n)


def identity(n: int) -> torch.Tensor:
    """
    Returns the nxn identity matrix.
    """
    return torch.eye(n)


def product(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return X @ Y


def hadamard(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return X * Y


def diag(X: torch.Tensor) -> torch.Tensor:
    return torch.unsqueeze(torch.diag(X), dim=1)


def Diag(x: torch.Tensor) -> torch.Tensor:
    return torch.diag(x.flatten())


def elements_sum(X: torch.Tensor):
    """
    Returns the sum of the elements of X.
    """
    return torch.sum(X)


def column_repeat(x: torch.Tensor, n: int) -> torch.Tensor:
    assert is_column_vector(x)
    return x.repeat(1, n)


def row_repeat(x: torch.Tensor, m: int) -> torch.Tensor:
    assert is_row_vector(x)
    return x.repeat(m, 1)


def columns_sum(X: torch.Tensor) -> torch.Tensor:
    return to_row(torch.sum(X, dim=0))


def rows_sum(X: torch.Tensor) -> torch.Tensor:
    return to_col(torch.sum(X, dim=1))


def columns_max(X: torch.Tensor) -> torch.Tensor:
    """
    Returns a column vector with the maximum values of each row in X.
    """
    return to_row(torch.max(X, dim=0)[0])


def rows_max(X: torch.Tensor) -> torch.Tensor:
    """
    Returns a row vector with the maximum values of each column in X.
    """
    return to_col(torch.max(X, dim=1)[0])


def columns_mean(X: torch.Tensor) -> torch.Tensor:
    """
    Returns a column vector with the mean values of each row in X.
    """
    return to_row(torch.mean(X, dim=0))


def rows_mean(X: torch.Tensor) -> torch.Tensor:
    """
    Returns a row vector with the mean values of each column in X.
    """
    return to_col(torch.mean(X, dim=1))


def apply(f, X: torch.Tensor) -> torch.Tensor:
    return f(X)


def exp(X: torch.Tensor) -> torch.Tensor:
    return torch.exp(X)


def log(X: torch.Tensor) -> torch.Tensor:
    return torch.log(X)


def reciprocal(X: torch.Tensor) -> torch.Tensor:
    return 1 / X


def square(X: torch.Tensor) -> torch.Tensor:
    return X * X


def sqrt(X: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(X)


def power_minus_half(X: torch.Tensor) -> torch.Tensor:
    return reciprocal(sqrt(X))
