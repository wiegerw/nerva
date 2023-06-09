# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import tensorflow as tf


def squared_error_loss(Y: tf.Tensor, T: tf.Tensor):
    return tf.reduce_sum(tf.pow(Y - T, 2))


def squared_error_loss_gradient(Y: tf.Tensor, T: tf.Tensor):
    return 2 * (Y - T)


def mean_squared_error_loss(Y, T):
    return tf.reduce_mean(tf.square(Y - T))


def mean_squared_error_loss_gradient(Y, T):
    return 2 * (Y - T) / tf.size(Y)
