# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import tensorflow as tf

def is_column_vector(x: tf.Tensor) -> bool:
    m, n = x.shape
    return n == 1


def is_row_vector(x: tf.Tensor) -> bool:
    m, n = x.shape
    return m == 1


def is_square(X: tf.Tensor) -> bool:
    m, n = X.shape
    return m == n


def dot(x, y):
    if is_column_vector(x) and is_column_vector(y):
        return tf.transpose(x) @ y
    elif is_row_vector(x) and is_row_vector(y):
        return x @ tf.transpose(y)
    raise RuntimeError('dot: received illegal input')


def to_row(x: tf.Tensor) -> tf.Tensor:
    if len(x.shape) == 1:
        return tf.expand_dims(x, axis=0)
    return x


def to_col(x: tf.Tensor) -> tf.Tensor:
    if len(x.shape) == 1:
        return tf.expand_dims(x, axis=1)
    return x


def zeros(m: int, n: int = 1) -> tf.Tensor:
    """
    Returns an mxn matrix with all elements equal to 0.
    """
    return tf.zeros([m, n], dtype=tf.float64)  # TODO: how to avoid hard coded types?


def ones(m: int, n: int = 1) -> tf.Tensor:
    """
    Returns an mxn matrix with all elements equal to 1.
    """
    return tf.ones([m, n], dtype=tf.float64)  # TODO: how to avoid hard coded types?


def identity(n: int) -> tf.Tensor:
    """
    Returns the nxn identity matrix.
    """
    return tf.eye(n, dtype=tf.float64)  # TODO: how to avoid hard coded types?


def product(X: tf.Tensor, Y: tf.Tensor) -> tf.Tensor:
    return X @ Y


def hadamard(X: tf.Tensor, Y: tf.Tensor) -> tf.Tensor:
    return X * Y


def diag(X: tf.Tensor) -> tf.Tensor:
    return tf.expand_dims(tf.linalg.diag_part(X), axis=1)


def Diag(x: tf.Tensor) -> tf.Tensor:
    return tf.linalg.diag(tf.reshape(x,[-1]))


def elements_sum(X: tf.Tensor):
    """
    Returns the sum of the elements of X.
    """
    return tf.reduce_sum(X)


def column_repeat(x: tf.Tensor, n: int) -> tf.Tensor:
    assert is_column_vector(x)
    return tf.tile(x, [1, n])


def row_repeat(x: tf.Tensor, m: int) -> tf.Tensor:
    assert is_row_vector(x)
    return tf.tile(x, [m, 1])


def columns_sum(X: tf.Tensor) -> tf.Tensor:
    return to_row(tf.reduce_sum(X, axis=0))


def rows_sum(X: tf.Tensor) -> tf.Tensor:
    return to_col(tf.reduce_sum(X, axis=1))


def columns_max(X: tf.Tensor) -> tf.Tensor:
    """
    Returns a column vector with the maximum values of each row in X.
    """
    return to_row(tf.reduce_max(X, axis=0))


def rows_max(X: tf.Tensor) -> tf.Tensor:
    """
    Returns a row vector with the maximum values of each column in X.
    """
    return to_col(tf.reduce_max(X, axis=1))


def columns_mean(X: tf.Tensor) -> tf.Tensor:
    """
    Returns a column vector with the mean values of each row in X.
    """
    return to_row(tf.reduce_mean(X, axis=0))


def rows_mean(X: tf.Tensor) -> tf.Tensor:
    """
    Returns a row vector with the mean values of each column in X.
    """
    return to_col(tf.reduce_mean(X, axis=1))


def apply(f, X: tf.Tensor) -> tf.Tensor:
    return f(X)


def exp(X: tf.Tensor) -> tf.Tensor:
    return tf.exp(X)


def log(X: tf.Tensor) -> tf.Tensor:
    return tf.math.log(X)


def inverse(X: tf.Tensor) -> tf.Tensor:
    return tf.math.reciprocal(X)


def square(X: tf.Tensor) -> tf.Tensor:
    return tf.math.square(X)


def sqrt(X: tf.Tensor) -> tf.Tensor:
    return tf.math.sqrt(X)


def power_minus_half(X: tf.Tensor) -> tf.Tensor:
    return inverse(sqrt(X))
