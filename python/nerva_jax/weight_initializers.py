# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import jax.numpy as jnp
import numpy as np

Matrix = jnp.ndarray


def set_bias_to_zero(b: Matrix):
    return jnp.zeros_like(b)


def set_weights_xavier(W: Matrix):
    K, D = W.shape
    xavier_stddev = np.sqrt(2 / (K + D))
    return jnp.array(np.random.randn(K, D) * xavier_stddev)


def set_bias_xavier(b: Matrix):
    return set_bias_to_zero(b)


def set_weights_xavier_normalized(W: Matrix):
    K, D = W.shape
    xavier_stddev = np.sqrt(2 / (K + D))
    return jnp.array(np.random.randn(K, D) * xavier_stddev)


def set_bias_xavier_normalized(b: Matrix):
    return set_bias_to_zero(b)


def set_weights_he(W: Matrix):
    K, D = W.shape
    he_stddev = np.sqrt(2 / D)
    random_matrix = np.random.randn(K, D)
    return jnp.array(random_matrix * he_stddev)


def set_bias_he(b: Matrix):
    return set_bias_to_zero(b)


def set_layer_weights(layer, text: str):
    if text == 'Xavier':
        layer.W = set_weights_xavier(layer.W)
        layer.b = set_bias_xavier(layer.b)
    elif text == 'XavierNormalized':
        layer.W = set_weights_xavier_normalized(layer.W)
        layer.b = set_bias_xavier_normalized(layer.b)
    elif text == 'He':
        layer.W = set_weights_he(layer.W)
        layer.b = set_bias_he(layer.b)
    else:
        raise RuntimeError(f'Could not parse weight initializer "{text}"')
