# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import numpy as np
import sympy as sp
from matrix_operations_sympy import zeros
from layers_sympy import LinearLayerColwise

def set_weights_xavier_sympy(W: sp.Matrix):
    K, D = W.shape
    xavier_stddev = sp.sqrt(2 / (K + D))
    random_matrix = np.random.randn(K, D)
    W[:, :] = sp.Matrix(random_matrix) * xavier_stddev


def set_weights_xavier_normalized_sympy(W: sp.Matrix):
    K, D = W.shape
    xavier_stddev = sp.sqrt(2 / (K + D))
    random_matrix = np.random.randn(K, D)
    W[:, :] = sp.Matrix(random_matrix) * xavier_stddev


def set_weights_he_sympy(W: sp.Matrix):
    K, D = W.shape
    he_stddev = sp.sqrt(2 / D)
    random_matrix = np.random.randn(K, D)
    W[:, :] = sp.Matrix(random_matrix) * he_stddev


def set_weights_xavier_numpy(W: np.ndarray):
    K, D = W.shape
    xavier_stddev = np.sqrt(2 / (K + D))
    W[:] = np.random.randn(K, D) * xavier_stddev


def set_weights_xavier_normalized_numpy(W: np.ndarray):
    K, D = W.shape
    xavier_stddev = np.sqrt(2 / (K + D))
    random_matrix = np.random.randn(K, D)
    W[:] = random_matrix * xavier_stddev


def set_weights_he_numpy(W: np.ndarray):
    K, D = W.shape
    he_stddev = np.sqrt(2 / D)
    random_matrix = np.random.randn(K, D)
    W[:] = random_matrix * he_stddev


def set_bias_to_zero(b: sp.Matrix):
    K, D = b.shape
    b[:, :] = zeros(K, D)


def set_weights(layer: LinearLayerColwise, text: str):
    if text == 'Xavier':
        set_weights_xavier_sympy(layer.W)
        set_bias_to_zero(layer.b)
    elif text == 'XavierNormalized':
        set_weights_xavier_normalized_sympy(layer.W)
        set_bias_to_zero(layer.b)
    elif text == 'He':
        set_weights_he_sympy(layer.W)
        set_bias_to_zero(layer.b)
    raise RuntimeError(f'Could not parse weight initializer "{text}"')
