# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import torch

Matrix = torch.Tensor


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
    return torch.dot(torch.squeeze(x), torch.squeeze(y))


def zeros(m: int, n=None) -> Matrix:
    """
    Returns an mxn matrix with all elements equal to 0.
    """
    return torch.zeros((m, n)) if n else torch.zeros(m)


def ones(m: int, n=None) -> Matrix:
    """
    Returns an mxn matrix with all elements equal to 1.
    """
    return torch.ones((m, n)) if n else torch.ones(m)


def identity(n: int) -> Matrix:
    """
    Returns the nxn identity matrix.
    """
    return torch.eye(n)


def product(X: Matrix, Y: Matrix) -> Matrix:
    return X @ Y


def hadamard(X: Matrix, Y: Matrix) -> Matrix:
    return X * Y


def diag(X: Matrix) -> Matrix:
    return torch.diag(X)


def Diag(x: Matrix) -> Matrix:
    return torch.diag(x.flatten())


def elements_sum(X: Matrix):
    """
    Returns the sum of the elements of X.
    """
    return torch.sum(X)


def column_repeat(x: Matrix, n: int) -> Matrix:
    assert is_column_vector(x)
    if len(x.shape) == 1:
        x = x.unsqueeze(1)
    return x.repeat(1, n)


def row_repeat(x: Matrix, m: int) -> Matrix:
    assert is_row_vector(x)
    if len(x.shape) == 1:
        x = x.unsqueeze(0)
    return x.repeat(m, 1)


def columns_sum(X: Matrix) -> Matrix:
    return torch.sum(X, dim=0)


def rows_sum(X: Matrix) -> Matrix:
    return torch.sum(X, dim=1)


def columns_max(X: Matrix) -> Matrix:
    """
    Returns a column vector with the maximum values of each row in X.
    """
    return torch.max(X, dim=0)[0]


def rows_max(X: Matrix) -> Matrix:
    """
    Returns a row vector with the maximum values of each column in X.
    """
    return torch.max(X, dim=1)[0]


def columns_mean(X: Matrix) -> Matrix:
    """
    Returns a column vector with the mean values of each row in X.
    """
    return torch.mean(X, dim=0)


def rows_mean(X: Matrix) -> Matrix:
    """
    Returns a row vector with the mean values of each column in X.
    """
    return torch.mean(X, dim=1)


def apply(f, X: Matrix) -> Matrix:
    return f(X)


def exp(X: Matrix) -> Matrix:
    return torch.exp(X)


def log(X: Matrix) -> Matrix:
    return torch.log(X)


def reciprocal(X: Matrix) -> Matrix:
    return 1 / X


def square(X: Matrix) -> Matrix:
    return X * X


def sqrt(X: Matrix) -> Matrix:
    return torch.sqrt(X)


def inv_sqrt(X: Matrix) -> Matrix:
    return 1 / torch.sqrt(X + epsilon)  # The epsilon is needed for numerical stability


def log_sigmoid(X: Matrix) -> Matrix:
    return -torch.nn.functional.softplus(-X)
