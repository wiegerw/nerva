# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import tensorflow as tf

Matrix = tf.Tensor

def is_vector(x: Matrix) -> bool:
    return len(x.shape) == 1


def is_square(X: Matrix) -> bool:
    m, n = X.shape
    return m == n


def dot(x, y):
    return tf.transpose(x) @ y


def zeros(m: int, n=None) -> Matrix:
    """
    Returns an mxn matrix with all elements equal to 0.
    """
    return tf.zeros([m, n], dtype=tf.float32) if n else tf.zeros([m], dtype=tf.float32)  # TODO: how to avoid hard coded types?


def ones(m: int, n=None) -> Matrix:
    """
    Returns an mxn matrix with all elements equal to 1.
    """
    return tf.ones([m, n], dtype=tf.float32) if n else tf.ones([m], dtype=tf.float32)  # TODO: how to avoid hard coded types?


def identity(n: int) -> Matrix:
    """
    Returns the nxn identity matrix.
    """
    return tf.eye(n, dtype=tf.float32)  # TODO: how to avoid hard coded types?


def product(X: Matrix, Y: Matrix) -> Matrix:
    return X @ Y


def hadamard(X: Matrix, Y: Matrix) -> Matrix:
    return X * Y


def diag(X: Matrix) -> Matrix:
    return tf.expand_dims(tf.linalg.diag_part(X), axis=1)


def Diag(x: Matrix) -> Matrix:
    return tf.linalg.diag(tf.reshape(x,[-1]))


def elements_sum(X: Matrix):
    """
    Returns the sum of the elements of X.
    """
    return tf.reduce_sum(X)


def column_repeat(x: Matrix, n: int) -> Matrix:
    assert is_vector(x)
    return tf.tile(tf.expand_dims(x, axis=1), [1, n])


def row_repeat(x: Matrix, m: int) -> Matrix:
    assert is_vector(x)
    return tf.tile(tf.expand_dims(x, axis=0), [m, 1])


def columns_sum(X: Matrix) -> Matrix:
    return tf.reduce_sum(X, axis=0)


def rows_sum(X: Matrix) -> Matrix:
    return tf.reduce_sum(X, axis=1)


def columns_max(X: Matrix) -> Matrix:
    """
    Returns a column vector with the maximum values of each row in X.
    """
    return tf.reduce_max(X, axis=0)


def rows_max(X: Matrix) -> Matrix:
    """
    Returns a row vector with the maximum values of each column in X.
    """
    return tf.reduce_max(X, axis=1)


def columns_mean(X: Matrix) -> Matrix:
    """
    Returns a column vector with the mean values of each row in X.
    """
    return tf.reduce_mean(X, axis=0)


def rows_mean(X: Matrix) -> Matrix:
    """
    Returns a row vector with the mean values of each column in X.
    """
    return tf.reduce_mean(X, axis=1)


def apply(f, X: Matrix) -> Matrix:
    return f(X)


def exp(X: Matrix) -> Matrix:
    return tf.exp(X)


def log(X: Matrix) -> Matrix:
    return tf.math.log(X)


def inverse(X: Matrix) -> Matrix:
    return tf.math.reciprocal(X)


def square(X: Matrix) -> Matrix:
    return tf.math.square(X)


def sqrt(X: Matrix) -> Matrix:
    return tf.math.sqrt(X)


def power_minus_half(X: Matrix) -> Matrix:
    return inverse(sqrt(X))
