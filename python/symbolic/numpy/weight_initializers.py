# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import numpy as np

Matrix = np.ndarray

def set_weights_xavier(W: Matrix):
    K, D = W.shape
    xavier_stddev = np.sqrt(2 / (K + D))
    W[:] = np.random.randn(K, D) * xavier_stddev


def set_weights_xavier_normalized(W: Matrix):
    K, D = W.shape
    xavier_stddev = np.sqrt(2 / (K + D))
    random_matrix = np.random.randn(K, D)
    W[:] = random_matrix * xavier_stddev


def set_weights_he(W: Matrix):
    K, D = W.shape
    he_stddev = np.sqrt(2 / D)
    random_matrix = np.random.randn(K, D)
    W[:] = random_matrix * he_stddev


def set_bias_to_zero(b: Matrix):
    K, D = b.shape
    b[:] = np.zeros(K, D)


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
