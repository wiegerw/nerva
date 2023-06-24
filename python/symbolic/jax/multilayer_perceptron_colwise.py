# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from nerva.datasets import load_dict_from_npz
from symbolic.jax.layers_colwise import *
from symbolic.jax.utilities import pp

Matrix = jnp.ndarray


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
                pp(f'b{index}', layer.b.T)  # TODO: avoid the transpose
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
                layer.W = data[f'W{index}']
                layer.b = data[f'b{index}']
                index += 1
