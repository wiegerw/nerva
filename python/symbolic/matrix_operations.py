# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from typing import List

import sympy as sp
from sympy import Matrix


def is_column_vector(x: Matrix) -> bool:
    m, n = x.shape
    return n == 1


def is_row_vector(x: Matrix) -> bool:
    m, n = x.shape
    return m == 1


def is_square(X: Matrix) -> bool:
    m, n = X.shape
    return m == n


def join_columns(columns: List[Matrix]) -> Matrix:
    assert all(is_column_vector(column) for column in columns)
    return Matrix([x.T for x in columns]).T


def join_rows(rows: List[Matrix]) -> Matrix:
    assert all(is_row_vector(row) for row in rows)
    return Matrix(rows)


def diff(f, X: Matrix) -> Matrix:
    """
    Returns the derivative of a matrix function
    :param f: a real-valued function
    :param X: a matrix
    :return: the derivative of f
    """
    m, n = X.shape
    return Matrix([[sp.diff(f, X[i, j]) for j in range(n)] for i in range(m)])


def substitute(expr, X: Matrix, Y: Matrix):
    assert X.shape == Y.shape
    m, n = X.shape
    substitutions = ((X[i, j], Y[i, j]) for i in range(m) for j in range(n))
    return expr.subs(substitutions)


def jacobian(x: Matrix, y) -> Matrix:
    assert is_column_vector(x) or is_row_vector(x)
    return x.jacobian(y)


def apply(f, x: Matrix) -> Matrix:
    return x.applyfunc(f)


def exp(x: Matrix) -> Matrix:
    return x.applyfunc(lambda x: sp.exp(x))


def log(x: Matrix) -> Matrix:
    return x.applyfunc(lambda x: sp.log(x))


def sqrt(x: Matrix) -> Matrix:
    return x.applyfunc(lambda x: sp.sqrt(x))


def inverse(x: Matrix) -> Matrix:
    return x.applyfunc(lambda x: 1 / x)


def power_minus_half(x: Matrix) -> Matrix:
    return inverse(sqrt(x))


def diag(X: Matrix) -> Matrix:
    assert is_square(X)
    m, n = X.shape
    return Matrix([[X[i, i] for i in range(m)]]).T


def Diag(x: Matrix) -> Matrix:
    assert is_column_vector(x) or is_row_vector(x)
    return sp.diag(*x)


def hadamard(x: Matrix, y: Matrix) -> Matrix:
    assert x.shape == y.shape
    m, n = x.shape
    return Matrix([[x[i, j] * y[i, j] for j in range(n)] for i in range(m)])
    # return matrix_multiply_elementwise(x, y)  ==> this may cause errors:
    #     def mul_elementwise(A, B):
    # >       assert A.domain == B.domain
    # E       AssertionError


def sum_columns(X: Matrix) -> Matrix:
    m, n = X.shape
    columns = [sum(X.col(j)) for j in range(n)]
    return Matrix(columns).T


def sum_rows(X: Matrix) -> Matrix:
    m, n = X.shape
    rows = [sum(X.row(i)) for i in range(m)]
    return Matrix(rows)


def repeat_column(x: Matrix, n: int) -> Matrix:
    assert is_column_vector(x)
    rows, cols = x.shape
    rows = [[x[i, 0]] * n for i in range(rows)]
    return Matrix(rows)


def repeat_row(x: Matrix, n: int) -> Matrix:
    assert is_row_vector(x)
    rows, cols = x.shape
    columns = [[x[0, j]] * n for j in range(cols)]
    return Matrix(columns).T


def rowwise_mean(X: Matrix) -> Matrix:
    m, n = X.shape
    return sum_rows(X) / n


def colwise_mean(X: Matrix) -> Matrix:
    m, n = X.shape
    return sum_columns(X) / m


def identity(n: int) -> Matrix:
    return sp.eye(n)


def ones(m: int, n: int) -> Matrix:
    return sp.ones(m, n)


def sum_elements(X: Matrix):
    m, n = X.shape

    return sum(X[i, j] for i in range(m) for j in range(n))
