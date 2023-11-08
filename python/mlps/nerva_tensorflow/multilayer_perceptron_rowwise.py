# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)
from typing import List

import tensorflow as tf

from mlps.nerva_tensorflow.layers_rowwise import BatchNormalizationLayer, LinearLayer, parse_linear_layer
from mlps.nerva_tensorflow.utilities import load_dict_from_npz, pp

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

    def info(self):
        index = 1
        for layer in self.layers:
            if isinstance(layer, LinearLayer):
                pp(f'W{index}', layer.W)
                pp(f'b{index}', layer.b)
                index += 1

    def load_weights_and_bias(self, filename: str):
        """
        Loads the weights and biases from a file in .npz format

        The weight matrices are stored using the keys W1, W2, ... and the bias vectors using the keys b1, b2, ...
        :param filename: the name of the file
        """
        print(f'Loading weights and bias from {filename}')
        data = load_dict_from_npz(filename)
        index = 1
        for layer in self.layers:
            if isinstance(layer, LinearLayer):
                layer.W.assign(data[f'W{index}'])
                layer.b.assign(data[f'b{index}'])
                index += 1


def parse_multilayer_perceptron(layer_specifications: List[str],
                                linear_layer_sizes: List[int],
                                optimizers: List[str],
                                linear_layer_weight_initializers: List[str]
                               ) -> MultilayerPerceptron:

    assert len(linear_layer_weight_initializers) == len(linear_layer_sizes) - 1
    layers = []

    linear_layer_index = 0
    optimizer_index = 0
    D = linear_layer_sizes[linear_layer_index]  # the input size of the current layer

    for specification in layer_specifications:
        if specification == 'BatchNormalization':
            layer = BatchNormalizationLayer(D)
            optimizer = optimizers[optimizer_index]
            layer.set_optimizer(optimizer)
            optimizer_index += 1
        else:
            K = linear_layer_sizes[linear_layer_index + 1]  # the output size of the layer
            optimizer = optimizers[optimizer_index]
            weight_initializer = linear_layer_weight_initializers[linear_layer_index]
            layer = parse_linear_layer(specification, D, K, optimizer, weight_initializer)
            optimizer_index += 1
            linear_layer_index += 1
            D = K
        layers.append(layer)
    return MultilayerPerceptron(layers)
