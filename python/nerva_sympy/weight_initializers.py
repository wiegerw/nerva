# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import numpy as np
import sympy as sp

from nerva_sympy.matrix_operations import zeros

Matrix = sp.Matrix


def set_bias_to_zero(b: Matrix):
    K, D = b.shape
    b[:] = zeros(K, D)


def set_weights_xavier(W: Matrix):
    K, D = W.shape
    xavier_stddev = sp.sqrt(2 / (K + D))
    random_matrix = np.random.randn(K, D)
    W[:] = Matrix(random_matrix) * xavier_stddev


def set_bias_xavier(b: Matrix):
    set_bias_to_zero(b)


def set_weights_xavier_normalized(W: Matrix):
    K, D = W.shape
    xavier_stddev = sp.sqrt(2 / (K + D))
    random_matrix = np.random.randn(K, D)
    W[:] = Matrix(random_matrix) * xavier_stddev


def set_bias_xavier_normalized(b: Matrix):
    set_bias_to_zero(b)


def set_weights_he(W: Matrix):
    K, D = W.shape
    he_stddev = sp.sqrt(2 / D)
    random_matrix = np.random.randn(K, D)
    W[:] = Matrix(random_matrix) * he_stddev


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
