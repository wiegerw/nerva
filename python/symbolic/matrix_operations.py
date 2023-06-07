# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from typing import List, Optional

import sympy as sp
from sympy import Matrix

def zeros(m: int, n: int = 1) -> Matrix:
    """
    Returns an mxn matrix with all elements equal to 0.
    """
    return sp.zeros(m, n)


def ones(m: int, n: int = 1) -> Matrix:
    """
    Returns an mxn matrix with all elements equal to 1.
    """
    return sp.ones(m, n)


def product(x: Matrix, y: Matrix) -> Matrix:
    return x * y


def hadamard(x: Matrix, y: Matrix) -> Matrix:
    assert x.shape == y.shape
    m, n = x.shape
    return Matrix([[x[i, j] * y[i, j] for j in range(n)] for i in range(m)])
    # return matrix_multiply_elementwise(x, y)  ==> this may cause errors:
    #     def mul_elementwise(A, B):
    # >       assert A.domain == B.domain
    # E       AssertionError


def identity(n: int) -> Matrix:
    """
    Returns the nxn identity matrix.
    """
    return sp.eye(n)


def diag(X: Matrix) -> Matrix:
    assert is_square(X)
    m, n = X.shape
    return Matrix([[X[i, i] for i in range(m)]]).T


def Diag(x: Matrix) -> Matrix:
    assert is_column_vector(x) or is_row_vector(x)
    return sp.diag(*x)


def elements_sum(X: Matrix):
    """
    Returns the sum of the elements of X.
    """
    m, n = X.shape
    return sum(X[i, j] for i in range(m) for j in range(n))


def column_repeat(x: Matrix, n: int) -> Matrix:
    assert is_column_vector(x)
    rows, cols = x.shape
    rows = [[x[i, 0]] * n for i in range(rows)]
    return Matrix(rows)


def row_repeat(x: Matrix, n: int) -> Matrix:
    assert is_row_vector(x)
    rows, cols = x.shape
    columns = [[x[0, j]] * n for j in range(cols)]
    return Matrix(columns).T


def columns_sum(X: Matrix) -> Matrix:
    m, n = X.shape
    columns = [sum(X.col(j)) for j in range(n)]
    return Matrix(columns).T


def rows_sum(X: Matrix) -> Matrix:
    m, n = X.shape
    rows = [sum(X.row(i)) for i in range(m)]
    return Matrix(rows)


def columns_max(x: Matrix) -> Matrix:
    """
    Returns a column vector with the maximum values of each row in X.
    """
    m, n = x.shape
    return Matrix([[sp.Max(*x.col(j)) for j in range(n)]])


def rows_max(X: Matrix) -> Matrix:
    """
    Returns a row vector with the maximum values of each column in X.
    """
    m, n = X.shape
    return Matrix([[sp.Max(*X.row(i)) for i in range(m)]]).T


def columns_mean(X: Matrix) -> Matrix:
    """
    Returns a column vector with the mean values of each row in X.
    """
    m, n = X.shape
    return columns_sum(X) / m


def rows_mean(X: Matrix) -> Matrix:
    """
    Returns a row vector with the mean values of each column in X.
    """
    m, n = X.shape
    return rows_sum(X) / n


def apply(f, x: Matrix) -> Matrix:
    return x.applyfunc(f)


def exp(x: Matrix) -> Matrix:
    return x.applyfunc(lambda x: sp.exp(x))


def log(x: Matrix) -> Matrix:
    return x.applyfunc(lambda x: sp.log(x))


def inverse(x: Matrix) -> Matrix:
    return x.applyfunc(lambda x: 1 / x)


def sqrt(x: Matrix) -> Matrix:
    return x.applyfunc(lambda x: sp.sqrt(x))


def power_minus_half(x: Matrix) -> Matrix:
    return inverse(sqrt(x))


###########################################################################################

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
