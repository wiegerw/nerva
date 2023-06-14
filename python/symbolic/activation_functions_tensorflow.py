# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import tensorflow as tf


def Relu(X: tf.Tensor):
    return tf.maximum(0.0, X)


def Relu_gradient(X: tf.Tensor):
    return tf.where(X > 0, tf.ones_like(X), tf.zeros_like(X))


def Leaky_relu(alpha):
    return lambda X: tf.maximum(alpha * X, X)


def Leaky_relu_gradient(alpha):
    return lambda X: tf.where(X > 0, tf.ones_like(X), tf.fill(tf.shape(X), tf.cast(alpha, X.dtype)))


def All_relu(alpha):
    return lambda X: tf.where(X < 0, tf.cast(alpha, X.dtype) * X, X)


def All_relu_gradient(alpha):
    return lambda X: tf.where(X < 0, tf.fill(tf.shape(X), tf.cast(alpha, X.dtype)), tf.ones_like(X))


def Hyperbolic_tangent(X: tf.Tensor):
    return tf.math.tanh(X)


def Hyperbolic_tangent_gradient(X: tf.Tensor):
    return 1 - tf.math.tanh(X) ** 2


def Sigmoid(X: tf.Tensor):
    return tf.math.sigmoid(X)


def Sigmoid_gradient(X: tf.Tensor):
    y = tf.math.sigmoid(X)
    return y * (1 - y)


def Srelu(al, tl, ar, tr):
    return lambda X: tf.where(X <= tl, tl + al * (X - tl),
                     tf.where(X < tr, X, tr + ar * (X - tr)))


def Srelu_gradient(al, tl, ar, tr):
    return lambda X: tf.where(X <= tl, al,
                     tf.where(X < tr, 1., ar))
