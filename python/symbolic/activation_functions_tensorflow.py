# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import tensorflow as tf


def relu(x):
    return tf.nn.relu(x)


def relu_prime(x):
    return tf.where(x > 0, tf.ones_like(x), tf.zeros_like(x))


def leaky_relu(alpha):
    return lambda x: tf.nn.leaky_relu(x, alpha)


def leaky_relu_prime(alpha):
    return lambda x: tf.where(x > 0, tf.ones_like(x), tf.fill(tf.shape(x), tf.cast(alpha, x.dtype)))


def all_relu(alpha):
    return lambda x: tf.where(x < 0, tf.cast(alpha, x.dtype) * x, x)


def all_relu_prime(alpha):
    return lambda x: tf.where(x < 0, tf.fill(tf.shape(x), tf.cast(alpha, x.dtype)), tf.ones_like(x))


def hyperbolic_tangent(x):
    return tf.math.tanh(x)


def hyperbolic_tangent_prime(x):
    return 1 - tf.math.tanh(x) ** 2


def sigmoid(x):
    return tf.math.sigmoid(x)


def sigmoid_prime(x):
    y = tf.math.sigmoid(x)
    return y * (1 - y)


def srelu(al, tl, ar, tr):
    return lambda x: tf.where(x <= tl, tl + al * (x - tl),
                     tf.where(x < tr, x, tr + ar * (x - tr)))


def srelu_prime(al, tl, ar, tr):
    return lambda x: tf.where(x <= tl, al,
                     tf.where(x < tr, 1., ar))
