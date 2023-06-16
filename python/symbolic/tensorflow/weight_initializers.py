# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import tensorflow as tf

Matrix = tf.Tensor

def set_weights_xavier(W: Matrix):
    initializer = tf.initializers.GlorotUniform()
    tf_w = tf.Variable(initializer(shape=W.shape))
    W.assign(tf_w)


def set_weights_xavier_normalized(W: Matrix):
    initializer = tf.initializers.GlorotNormal()
    tf_w = tf.Variable(initializer(shape=W.shape))
    W.assign(tf_w)


def set_weights_he(W: Matrix):
    initializer = tf.initializers.HeNormal()
    tf_w = tf.Variable(initializer(shape=W.shape))
    W.assign(tf_w)


def set_bias_to_zero(b: Matrix):
    b.assign(tf.zeros_like(b))


def set_weights(layer, text: str):
    if text == 'Xavier':
        set_weights_xavier(layer.W)
        set_bias_to_zero(layer.b)
    elif text == 'XavierNormalized':
        set_weights_xavier_normalized(layer.W)
        set_bias_to_zero(layer.b)
    elif text == 'He':
        set_weights_he(layer.W)
        set_bias_to_zero(layer.b)
    raise RuntimeError(f'Could not parse weight initializer "{text}"')
