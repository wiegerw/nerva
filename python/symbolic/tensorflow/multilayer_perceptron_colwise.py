# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from typing import List

from symbolic.tensorflow.layers_colwise import *

Matrix = tf.Tensor


class MultilayerPerceptron(object):
    """
    Multilayer perceptron
    """
    def __init__(self, layers=None):
        if not layers:
            layers = []
        self.layers = layers

    def feedforward(self, X: Matrix) -> Matrix:
        for layer in self.layers:
            X = layer.feedforward(X)
        return X

    def backpropagate(self, Y: Matrix, DY: Matrix) -> None:
        for layer in reversed(self.layers):
            layer.backpropagate(Y, DY)
            Y, DY = layer.X, layer.DX

    def optimize(self, eta: float):
        for layer in self.layers:
            layer.optimize(eta)


def parse_multilayer_perceptron(layer_specifications: List[str],
                                linear_layer_sizes: List[int],
                                linear_layer_activations: List[str],
                                linear_layer_optimizers: List[str],
                                linear_layer_weight_initializers: List[str],
                                batch_size: int
                               ) -> MultilayerPerceptron:

    assert len(linear_layer_activations) == len(linear_layer_optimizers) == len(linear_layer_weight_initializers) == len(linear_layer_sizes) - 1
    layers = []

    linear_layer_index = 0
    D = linear_layer_sizes[linear_layer_index]  # the input size of the current layer
    N = batch_size

    for specification in layer_specifications:
        if specification == 'BatchNormalization':
            layer = BatchNormalizationLayer(D, N)
        else:
            K = linear_layer_sizes[linear_layer_index + 1]  # the output size of the layer
            optimizer = linear_layer_optimizers[linear_layer_index]
            weight_initializer = linear_layer_weight_initializers[linear_layer_index]
            layer = parse_linear_layer(specification, D, K, N, optimizer, weight_initializer)
            linear_layer_index += 1
            D = K
        layers.append(layer)

    return MultilayerPerceptron(layers)
