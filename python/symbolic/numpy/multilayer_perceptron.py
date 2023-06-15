# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import re

from symbolic.numpy.layers import *
from symbolic.numpy.optimizers import *
from symbolic.numpy.weight_initializers import set_weights

Matrix = np.Matrix


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


def parse_optimizer(text: str,
                    layer: LinearLayerColwise
                   ) -> Optimizer:
    try:
        if text == 'GradientDescent':
            return GradientDescentOptimizer(layer.W, layer.DW, layer.b, layer.Db)
        elif text.startswith('Momentum'):
            m = re.match(r'Momentum\((.*)\)$', text)
            mu = float(m.group(1))
            return MomentumOptimizer(layer.W, layer.DW, layer.b, layer.Db, mu)
        elif text.startswith('Nesterov'):
            m = re.match(r'Nesterov\((.*)\)$', text)
            mu = float(m.group(1))
            return NesterovOptimizer(layer.W, layer.DW, layer.b, layer.Db, mu)
    except:
        pass
    raise RuntimeError(f'Could not parse optimizer "{text}"')


def parse_srelu_activation(text: str) -> SReLUActivation:
    try:
        if text == 'SReLU':
            return SReLUActivation()
        else:
            m = re.match(r'SReLU\(([^,]*),([^,]*),([^,]*),([^,]*)\)$', text)
            al = float(m.group(1))
            tl = float(m.group(2))
            ar = float(m.group(3))
            tr = float(m.group(4))
            return SReLUActivation(al, tl, ar, tr)
    except:
        pass
    raise RuntimeError(f'Could not parse SReLU activation "{text}"')


def parse_activation(text: str) -> ActivationFunction:
    try:
        if text == 'ReLU':
            return ReLUActivation()
        elif text == 'HyperbolicTangent':
            return HyperbolicTangentActivation()
        elif text.startswith('AllReLU'):
            m = re.match(r'AllReLU\((.*)\)$', text)
            alpha = float(m.group(1))
            return AllReLUActivation(alpha)
        elif text.startswith('LeakyReLU'):
            m = re.match(r'LeakyReLU\((.*)\)$', text)
            alpha = float(m.group(1))
            return LeakyReLUActivation(alpha)
    except:
        pass
    raise RuntimeError(f'Could not parse activation "{text}"')


def parse_linear_layer(text: str,
                       D: int,
                       K: int,
                       N: int,
                       optimizer: str,
                       weight_initializer: str
                      ) -> Layer:
    if text == 'Linear':
        layer = LinearLayerColwise(D, K, N)
    elif text == 'Sigmoid':
        layer = SigmoidLayerColwise(D, K, N)
    elif text == 'Softmax':
        layer = SoftmaxLayerColwise(D, K, N)
    elif text == 'LogSoftmax':
        layer = LogSoftmaxLayerColwise(D, K, N)
    elif text.startswith('SReLU'):
        act = parse_srelu_activation(text)
        layer = SReLULayerColwise(D, K, N, act)
    else:
        act = parse_activation(text)
        layer = ActivationLayerColwise(D, K, N, act)
    layer.optimizer = parse_optimizer(optimizer, layer)
    set_weights(layer, weight_initializer)
    return layer


def create_multilayer_perceptron(layer_specifications: List[str],
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
            layer = BatchNormalizationLayerColwise(D, N)
        else:
            K = linear_layer_sizes[linear_layer_index + 1]  # the output size of the layer
            optimizer = linear_layer_optimizers[linear_layer_index]
            weight_initializer = linear_layer_weight_initializers[linear_layer_index]
            layer = parse_linear_layer(specification, D, K, N, optimizer, weight_initializer)
            linear_layer_index += 1
            D = K
        layers.append(layer)

    return MultilayerPerceptron(layers)
