# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from symbolic.layers_sympy import *

class MultilayerPerceptron(object):

    def __init__(self):
        self.layers = []

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
