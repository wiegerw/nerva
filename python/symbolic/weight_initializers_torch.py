# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import torch
from layers_torch import LinearLayerColwise


def set_weights_xavier(W: torch.Tensor):
    K, D = W.shape
    xavier_stddev = torch.sqrt(torch.tensor(2.0 / (K + D)))
    W.data = torch.randn(K, D) * xavier_stddev


def set_weights_xavier_normalized(W: torch.Tensor):
    K, D = W.shape
    xavier_stddev = torch.sqrt(torch.tensor(2.0 / (K + D)))
    random_matrix = torch.randn(K, D)
    W.data = random_matrix * xavier_stddev


def set_weights_he(W: torch.Tensor):
    K, D = W.shape
    he_stddev = torch.sqrt(torch.tensor(2.0 / D))
    random_matrix = torch.randn(K, D)
    W.data = random_matrix * he_stddev


def set_bias_to_zero(b: torch.Tensor):
    b.data.zero_()


def set_weights(layer: LinearLayerColwise, text: str):
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
