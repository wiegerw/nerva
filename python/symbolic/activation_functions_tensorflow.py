# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import tensorflow as tf


def relu(x):
    return tf.nn.relu(x)


def relu_derivative(x):
    return tf.where(x > 0, tf.ones_like(x), tf.zeros_like(x))


def leaky_relu(alpha):
    return lambda x: tf.nn.leaky_relu(x, alpha)


def leaky_relu_derivative(alpha):
    return lambda x: tf.where(x > 0, tf.ones_like(x), tf.fill(tf.shape(x), alpha))


def all_relu(alpha):
    return lambda x: tf.where(x < 0, alpha * x, x)


def all_relu_derivative(alpha):
    return lambda x: tf.where(x < 0, tf.fill(tf.shape(x), alpha), tf.ones_like(x))


def hyperbolic_tangent(x):
    return tf.math.tanh(x)


def hyperbolic_tangent_derivative(x):
    return 1 - tf.math.tanh(x) ** 2


def sigmoid(x):
    return tf.math.sigmoid(x)


def sigmoid_derivative(x):
    y = tf.math.sigmoid(x)
    return y * (1 - y)


# TODO: check this
def srelu(al, tl, ar, tr):
    return lambda x: tf.where(x <= tl, tl + al * (x - tl),
                    tf.where(x >= tr, tr + ar * (x - tr), x))


# TODO: check this
def srelu_derivative(al, tl, ar, tr):
    return lambda x: tf.where((x <= tl) | (x >= tr), tf.zeros_like(x),
                     tf.where((tl < x) & (x < tr), tf.ones_like(x),
                              tf.where(x < tl, tf.fill(tf.shape(x), al), tf.fill(tf.shape(x), ar))))
