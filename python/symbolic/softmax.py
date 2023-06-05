# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from symbolic.matrix_operations import *

#-------------------------------------#
#           softmax colwise
#-------------------------------------#

def softmax_colwise(X: Matrix) -> Matrix:
    D, N = X.shape
    E = exp(X)
    return hadamard(E, repeat_row(inverse(sum_columns(E)), D))


def softmax_colwise1(X: Matrix) -> Matrix:
    D, N = X.shape

    def softmax(x):
        e = exp(x)
        return e / sum(e)

    return Matrix([softmax(X.col(j)).T for j in range(N)]).T


def stable_softmax_colwise(X: Matrix) -> Matrix:
    D, N = X.shape
    c = Matrix(sp.symarray('C', (1, N), real=True))
    E = exp(X - repeat_row(c, D))
    return hadamard(E, repeat_row(inverse(sum_columns(E)), D))


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
    D, N = X.shape
    return X - repeat_row(log(sum_columns(exp(X))), D)


def log_softmax_colwise1(X: Matrix) -> Matrix:
    return log(softmax_colwise(X))


def stable_log_softmax_colwise(X: Matrix) -> Matrix:
    D, N = X.shape
    c = Matrix(sp.symarray('C', (1, N), real=True))
    Y = X - repeat_row(c, D)
    return Y - repeat_row(log(sum_columns(exp(Y))), D)


def log_softmax_colwise_derivative(x: Matrix) -> Matrix:
    assert is_column_vector(x)
    D, N = x.shape
    return sp.eye(D) - repeat_row(softmax_colwise(x).T, D)


def log_softmax_colwise_derivative1(x: Matrix) -> Matrix:
    return jacobian(log_softmax_colwise(x), x)


#-------------------------------------#
#           softmax rowwise
#-------------------------------------#

def softmax_rowwise(X: Matrix) -> Matrix:
    N, D = X.shape
    E = exp(X)
    return hadamard(E, repeat_column(inverse(sum_rows(E)), D))


def softmax_rowwise1(X: Matrix) -> Matrix:
    N, D = X.shape

    def softmax(x):
        e = exp(x)
        return e / sum(e)

    return join_rows([softmax(X.row(i)) for i in range(N)])


def softmax_rowwise2(X: Matrix) -> Matrix:
    return softmax_colwise(X.T).T


def stable_softmax_rowwise(X: Matrix) -> Matrix:
    N, D = X.shape
    c = Matrix(sp.symarray('C', (N, 1), real=True))
    E = exp(X - repeat_column(c, D))
    return hadamard(E, repeat_column(inverse(sum_rows(E)), D))


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
    N, D = X.shape
    return X - repeat_column(log(sum_rows(exp(X))), D)


def log_softmax_rowwise1(X: Matrix) -> Matrix:
    return log(softmax_rowwise(X))


def log_softmax_rowwise2(X: Matrix) -> Matrix:
    return log_softmax_colwise(X.T).T


def stable_log_softmax_rowwise(X: Matrix) -> Matrix:
    N, D = X.shape
    c = Matrix(sp.symarray('C', (N, 1), real=True))
    Y = X - repeat_column(c, D)
    return Y - repeat_column(log(sum_rows(exp(Y))), D)


def log_softmax_rowwise_derivative(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    N, D = x.shape
    return sp.eye(D) - repeat_row(softmax_rowwise(x), D)


def log_softmax_rowwise_derivative1(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    return jacobian(log_softmax_rowwise(x), x)


def log_softmax_rowwise_derivative2(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    return log_softmax_colwise_derivative(x.T)
