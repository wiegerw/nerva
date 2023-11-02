# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import torch

Matrix = torch.Tensor

def set_bias_to_zero(b: Matrix):
    b.data.zero_()


def set_weights_xavier(W: Matrix):
    K, D = W.shape
    xavier_stddev = torch.sqrt(torch.tensor(2.0 / (K + D)))
    W.data = torch.randn(K, D) * xavier_stddev


def set_bias_xavier(b: Matrix):
    set_bias_to_zero(b)


def set_weights_xavier_normalized(W: Matrix):
    K, D = W.shape
    xavier_stddev = torch.sqrt(torch.tensor(2.0 / (K + D)))
    random_matrix = torch.randn(K, D)
    W.data = random_matrix * xavier_stddev


def set_bias_xavier_normalized(b: Matrix):
    set_bias_to_zero(b)


def set_weights_he(W: Matrix):
    K, D = W.shape
    he_stddev = torch.sqrt(torch.tensor(2.0 / D))
    random_matrix = torch.randn(K, D)
    W.data = random_matrix * he_stddev


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
