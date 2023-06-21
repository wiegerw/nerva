# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

Matrix = tf.Tensor

def set_bias_to_zero(b: Matrix):
    b.assign(tf.zeros_like(b))


def set_weights_xavier(W: Matrix):
    initializer = tf.initializers.GlorotUniform()
    tf_w = tf.Variable(initializer(shape=W.shape))
    W.assign(tf.cast(tf_w, dtype=W.dtype))

def set_bias_xavier(b: Matrix):
    set_bias_to_zero(b)


def set_weights_xavier_normalized(W: Matrix):
    initializer = tf.initializers.GlorotNormal()
    tf_w = tf.Variable(initializer(shape=W.shape))
    W.assign(tf.cast(tf_w, dtype=W.dtype))


def set_bias_xavier_normalized(b: Matrix):
    set_bias_to_zero(b)


def set_weights_he(W: Matrix):
    initializer = tf.initializers.HeNormal()
    tf_w = tf.Variable(initializer(shape=W.shape))
    W.assign(tf.cast(tf_w, dtype=W.dtype))


def set_bias_he(b: Matrix):
    set_bias_to_zero(b)


def set_layer_weights(layer, text: str):
    if text == 'Xavier':
        set_weights_xavier(layer.W)
        set_bias_xavier(layer.b)
    elif text == 'XavierNormalized':
        set_weights_xavier_normalized(layer.W)
        set_bias_xavier_normalized(layer.b)
    elif text == 'He':
        set_weights_he(layer.W)
        set_bias_he(layer.b)
    else:
        raise RuntimeError(f'Could not parse weight initializer "{text}"')
