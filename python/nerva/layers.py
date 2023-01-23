# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from typing import Optional, List, Union, Tuple
from nerva.activation import Activation, NoActivation, ReLU, AllReLU, Softmax, Sigmoid, HyperbolicTangent, LeakyReLU
from nerva.optimizers import Optimizer, GradientDescent
from nerva.utilities import RandomNumberGenerator
from nerva.weights import Weights
import nervalib


class Layer(object):
    pass


class Dense(Layer):
    def __init__(self,
                 units: int,
                 activation: Activation=NoActivation(),
                 optimizer: Optimizer=GradientDescent(),
                 weight_initializer: Weights=Weights.Xavier
                ):
        """
        A dense layer.

        :param units: the number of outputs of the layer
        :param activation: the activation function
        :param optimizer: the optimizer
        :param weight_initializer: the weight initializer
        """
        self.units = int(units) if not isinstance(units, int) else units
        if self.units < 0:
            raise ValueError(
                f"Received an invalid value for `units`, expected "
                f"a positive integer. Received: units={units}"
            )
        self.activation = activation
        self.optimizer = optimizer
        self.weight_initializer = weight_initializer
        self.input_size = -1

    def compile(self, rng: RandomNumberGenerator, batch_size: int, dropout_rate: float=0.0):
        """
        Compiles the model into a C++ object

        :param rng: a random number generator
        :param batch_size: the batch size
        :param dropout_rate: the dropout rate
        :return:
        """
        layer = None
        if dropout_rate == 0.0:
            if isinstance(self.activation, NoActivation):
                layer = nervalib.linear_layer(self.input_size, self.units, batch_size)
            elif isinstance(self.activation, ReLU):
                layer = nervalib.relu_layer(self.input_size, self.units, batch_size)
            elif isinstance(self.activation, AllReLU):
                layer = nervalib.all_relu_layer(self.activation.alpha, self.input_size, self.units, batch_size)
            elif isinstance(self.activation, LeakyReLU):
                layer = nervalib.leaky_relu_layer(self.activation.alpha, self.input_size, self.units, batch_size)
            elif isinstance(self.activation, Sigmoid):
                layer = nervalib.sigmoid_layer(self.input_size, self.units, batch_size)
            elif isinstance(self.activation, Softmax):
                layer = nervalib.softmax_layer(self.input_size, self.units, batch_size)
            elif isinstance(self.activation, HyperbolicTangent):
                layer = nervalib.hyperbolic_tangent_layer(self.input_size, self.units, batch_size)
        else:
            if isinstance(self.activation, NoActivation):
                layer = nervalib.linear_dropout_layer(self.input_size, self.units, batch_size, dropout_rate)
            elif isinstance(self.activation, ReLU):
                layer = nervalib.relu_dropout_layer(self.input_size, self.units, batch_size, dropout_rate)
            elif isinstance(self.activation, AllReLU):
                layer = nervalib.all_relu_dropout_layer(self.activation.alpha, self.input_size, self.units, batch_size, dropout_rate)
            elif isinstance(self.activation, LeakyReLU):
                layer = nervalib.leaky_relu_dropout_layer(self.activation.alpha, self.input_size, self.units, batch_size, dropout_rate)
            elif isinstance(self.activation, Sigmoid):
                layer = nervalib.sigmoid_dropout_layer(self.input_size, self.units, batch_size, dropout_rate)
            elif isinstance(self.activation, Softmax):
                layer = nervalib.softmax_dropout_layer(self.input_size, self.units, batch_size, dropout_rate)
            elif isinstance(self.activation, HyperbolicTangent):
                layer = nervalib.hyperbolic_tangent_dropout_layer(self.input_size, self.units, batch_size, dropout_rate)

        if not layer:
            raise RuntimeError('Unsupported layer type')

        layer.initialize_weights(self.weight_initializer, rng)

        return layer


class Sparse(Layer):
    def __init__(self, units: int, sparsity: float, activation: Activation=NoActivation(), optimizer=GradientDescent(), weight_initializer=Weights.Xavier):
        """
        A sparse layer.

        :param units: the number of outputs of the layer
        :param sparsity: the sparsity of the layer. This is a number between 0.0 (fully dense) and 1.0 (fully sparse).
        :param activation: the activation function
        :param optimizer: the optimizer
        """
        self.units = int(units) if not isinstance(units, int) else units
        if self.units < 0:
            raise ValueError(
                f"Received an invalid value for `units`, expected "
                f"a positive integer. Received: units={units}"
            )
        self.sparsity = sparsity
        self.activation = activation
        self.optimizer = optimizer
        self.weight_initializer = weight_initializer
        self.input_size = -1
        self._layer = None

    def compile(self, rng: RandomNumberGenerator, batch_size: int, dropout_rate: float=0.0):
        """
        Compiles the model into a C++ object

        :param rng: a random number generator
        :param batch_size: the batch size
        :param dropout_rate: the dropout rate
        :return:
        """
        layer = None
        if dropout_rate == 0.0:
            if isinstance(self.activation, NoActivation):
                layer = nervalib.sparse_linear_layer(self.input_size, self.units, batch_size, self.sparsity, rng)
            elif isinstance(self.activation, ReLU):
                layer = nervalib.sparse_relu_layer(self.input_size, self.units, batch_size, self.sparsity, rng)
            elif isinstance(self.activation, Sigmoid):
                layer = nervalib.sparse_sigmoid_layer(self.input_size, self.units, batch_size, self.sparsity, rng)
            elif isinstance(self.activation, Softmax):
                layer = nervalib.sparse_softmax_layer(self.input_size, self.units, batch_size, self.sparsity, rng)
            elif isinstance(self.activation, AllReLU):
                layer = nervalib.sparse_all_relu_layer(self.activation.alpha, self.input_size, self.units, batch_size, self.sparsity, rng)
            elif isinstance(self.activation, HyperbolicTangent):
                layer = nervalib.sparse_hyperbolic_tangent_layer(self.input_size, self.units, batch_size, self.sparsity, rng)

        if not layer:
            raise RuntimeError('Unsupported layer type')

        layer.initialize_weights(self.weight_initializer, rng)
        self._layer = layer
        return layer

    def regrow(self, weight_initializer: Weights, zeta: float, rng: RandomNumberGenerator):
        """Prunes and regrows the weights

        :param weight_initializer: A weight initializer
        :param zeta: The fraction of weights that is regrown
        :param rng: A random number generator
        """
        assert self._layer
        nervalib.regrow_sparse_layer(self._layer, weight_initializer, zeta, rng)


class Dropout(Layer):
    def __init__(self, rate: float):
        self.rate = rate


class BatchNormalization(Layer):
    def __init__(self):
        self.input_size = -1

    def compile(self, batch_size: int):
        return nervalib.batch_normalization_layer(self.input_size, batch_size)


class SimpleBatchNormalization(Layer):
    def __init__(self):
        self.input_size = -1

    def compile(self, batch_size: int):
        return nervalib.simple_batch_normalization_layer(self.input_size, batch_size)


class AffineTransform(Layer):
    def __init__(self):
        self.input_size = -1

    def compile(self, batch_size: int):
        return nervalib.affine_layer(self.input_size, batch_size)


# neural networks
class Sequential(object):
    def __init__(self, layers: Optional[Union[List[Layer], Tuple[Layer]]]=None):
        self.layers = []
        self.compiled_model = None
        if layers:
            for layer in layers:
                self.add(layer)

    def add(self, layer: Layer):
        self.layers.append(layer)

    def _check_layers(self):
        """
        Checks if the architecture of the layers is OK
        """
        layers = self.layers

        # At least one layer
        if not layers:
            raise RuntimeError('No layers are defined')

        # Dropout may only appear after Dense
        for i, layer in enumerate(layers):
            if isinstance(layer, Dropout):
                if i == 0:
                    raise RuntimeError('The first layer cannot be a dropout layer')
                if not isinstance(layers[i-1], Dense):
                    raise RuntimeError(f'Dropout layer {i} is not preceded by a Dense layer')

    def compile(self, input_size: int, batch_size: int, rng: RandomNumberGenerator) -> None:
        self._check_layers()

        M = nervalib.MLP()

        # add layers
        n = len(self.layers)
        input_size = input_size  # keep track of the input size, since it isn't stored in the layers
        for i, layer in enumerate(self.layers):
            if isinstance(layer, (Dense, Sparse)):
                layer.input_size = input_size
                input_size = layer.units
                dropout_rate = 0.0
                if i + 1 < n and isinstance(self.layers[i+1], Dropout):
                    dropout_rate = self.layers[i+1].rate
                cpp_layer = layer.compile(rng, batch_size, dropout_rate)
                cpp_layer.set_optimizer(layer.optimizer.compile())
                cpp_layer.initialize_weights(layer.weight_initializer, rng)
                M.append_layer(cpp_layer)
            elif isinstance(layer, (BatchNormalization, SimpleBatchNormalization, AffineTransform)):
                layer.input_size = input_size
                cpp_layer = layer.compile(batch_size)
                M.append_layer(cpp_layer)
        self.compiled_model = M

    def regrow(self, weight_initializer: Weights, zeta: float, rng: RandomNumberGenerator):
        """Prunes and regrows the weights of the sparse layers

        :param weight_initializer: A weight initializer
        :param zeta: The fraction of weights that is regrown
        :param rng: A random number generator
        """
        for layer in self.layers:
            if isinstance(layer, Sparse):
                layer.regrow(weight_initializer, zeta, rng)

    def feedforward(self, X):
        return self.compiled_model.feedforward(X)

    def backpropagate(self, Y, dY):
        return self.compiled_model.backpropagate(Y, dY)

    def optimize(self, eta):
        self.compiled_model.optimize(eta)
