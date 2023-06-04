# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from symbolic.matrix_operations import *

#-------------------------------------#
#           softmax colwise
#-------------------------------------#

def softmax_colwise(X: Matrix) -> Matrix:
    m, n = X.shape
    E = exp(X)
    return hadamard(E, repeat_row(inverse(sum_columns(E)), m))


def softmax_colwise1(X: Matrix) -> Matrix:
    m, n = X.shape

    def softmax(x):
        e = exp(x)
        return e / sum(e)

    return Matrix([softmax(X.col(j)).T for j in range(n)]).T


def stable_softmax_colwise(X: Matrix) -> Matrix:
    m, n = X.shape
    c = Matrix(sp.symarray('C', (1, n), real=True))
    E = exp(X - repeat_row(c, m))
    return hadamard(E, repeat_row(inverse(sum_columns(E)), m))


def softmax_colwise_derivative(x: Matrix) -> Matrix:
    assert is_column_vector(x)
    y = softmax_colwise1(x)
    return Diag(y) - y * y.T


def softmax_colwise_derivative1(x: Matrix) -> Matrix:
    return jacobian(softmax_colwise1(x), x)


#-------------------------------------#
#           log_softmax colwise
#-------------------------------------#

def log_softmax_colwise(X: Matrix) -> Matrix:
    m, n = X.shape
    return X - repeat_row(log(sum_columns(exp(X))), m)


def log_softmax_colwise1(X: Matrix) -> Matrix:
    return log(softmax_colwise(X))


def stable_log_softmax_colwise(X: Matrix) -> Matrix:
    m, n = X.shape
    c = Matrix(sp.symarray('C', (1, n), real=True))
    Y = X - repeat_row(c, m)
    return Y - repeat_row(log(sum_columns(exp(Y))), m)


def log_softmax_colwise_derivative(x: Matrix) -> Matrix:
    assert is_column_vector(x)
    m, n = x.shape
    return sp.eye(m) - repeat_row(softmax_colwise(x).T, m)


def log_softmax_colwise_derivative1(x: Matrix) -> Matrix:
    return jacobian(log_softmax_colwise(x), x)


#-------------------------------------#
#           softmax rowwise
#-------------------------------------#

def softmax_rowwise(X: Matrix) -> Matrix:
    m, n = X.shape
    E = exp(X)
    return hadamard(E, repeat_column(inverse(sum_rows(E)), n))


def softmax_rowwise1(X: Matrix) -> Matrix:
    m, n = X.shape

    def softmax(x):
        e = exp(x)
        return e / sum(e)

    return join_rows([softmax(X.row(i)) for i in range(m)])


def softmax_rowwise2(X: Matrix) -> Matrix:
    return softmax_colwise(X.T).T


def stable_softmax_rowwise(X: Matrix) -> Matrix:
    m, n = X.shape
    c = Matrix(sp.symarray('C', (m, 1), real=True))
    E = exp(X - repeat_column(c, n))
    return hadamard(E, repeat_column(inverse(sum_rows(E)), n))


def softmax_rowwise_derivative(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    y = softmax_rowwise(x)
    return Diag(y) - y.T * y


def softmax_rowwise_derivative1(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    return jacobian(softmax_rowwise(x), x)


def softmax_rowwise_derivative2(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    return softmax_colwise_derivative(x.T).T


#-------------------------------------#
#           log_softmax rowwise
#-------------------------------------#

def log_softmax_rowwise(X: Matrix) -> Matrix:
    m, n = X.shape
    return X - repeat_column(log(sum_rows(exp(X))), n)


def log_softmax_rowwise1(X: Matrix) -> Matrix:
    return log(softmax_rowwise(X))


def log_softmax_rowwise2(X: Matrix) -> Matrix:
    return log_softmax_colwise(X.T).T


def stable_log_softmax_rowwise(X: Matrix) -> Matrix:
    m, n = X.shape
    c = Matrix(sp.symarray('C', (m, 1), real=True))
    Y = X - repeat_column(c, n)
    return Y - repeat_column(log(sum_rows(exp(Y))), n)


def log_softmax_rowwise_derivative(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    m, n = x.shape
    return sp.eye(n) - repeat_row(softmax_rowwise(x), n)


def log_softmax_rowwise_derivative1(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    return jacobian(log_softmax_rowwise(x), x)


def log_softmax_rowwise_derivative2(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    return log_softmax_colwise_derivative(x.T)
