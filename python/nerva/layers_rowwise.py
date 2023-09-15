# Copyright 2022 - 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from typing import Optional, List, Union, Tuple

from nerva.activation import Activation, NoActivation
from nerva.optimizers import Optimizer, GradientDescent
from nerva.weights import WeightInitializer, Xavier
import nervalibrowwise


class Layer(object):
    pass


def print_activation(activation: Activation) -> str:
    if isinstance(activation, NoActivation):
        return 'Linear'
    return str(activation).replace('()', '')


class Dense(Layer):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 activation: Activation=NoActivation(),
                 optimizer: Optimizer=GradientDescent(),
                 weight_initializer: WeightInitializer=Xavier(),
                 dropout_rate: float=0
                ):
        """
        A dense layer.

        :param input_size: the number of inputs of the layer
        :param output_size: the number of outputs of the layer
        :param activation: the activation function
        :param optimizer: the optimizer
        :param weight_initializer: the weight initializer
        :param dropout_rate: the dropout rate
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.optimizer = optimizer
        self.weight_initializer = weight_initializer
        self.dropout_rate = dropout_rate
        self._layer = None

    def __str__(self):
        return f'Dense(output_size={self.output_size}, activation={self.activation}, optimizer={self.optimizer}, weight_initializer={self.weight_initializer}, dropout={self.dropout_rate})'

    def density_info(self) -> str:
        N = self._layer.W.size
        return f'{N}/{N} (100%)'

    def set_weights_and_bias(self, init: WeightInitializer) -> None:
        self._layer.set_weights_and_bias(init.compile())

    def compile(self, batch_size: int):
        """
        Compiles the model into a C++ object

        :param batch_size: the batch size
        :return:
        """
        activation = print_activation(self.activation)
        if self.dropout_rate == 0.0:
            layer = nervalibrowwise.make_dense_linear_layer(self.input_size, self.output_size, batch_size, activation, self.weight_initializer.compile(), self.optimizer.compile())
        else:
            layer = nervalibrowwise.make_dense_linear_dropout_layer(self.input_size, self.output_size, batch_size, self.dropout_rate, activation, self.weight_initializer.compile(), self.optimizer.compile())
        self._layer = layer
        return layer


class Sparse(Layer):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 density: float,
                 activation: Activation=NoActivation(),
                 optimizer=GradientDescent(),
                 weight_initializer=Xavier()):
        """
        A sparse layer.

        :param input_size: the number of inputs of the layer
        :param output_size: the number of outputs of the layer
        :param density: a hint for the maximum density of the layer. This is a number between 0.0 (fully sparse) and
         1.0 (fully dense). Memory will be reserved to store a matrix with the given density.
        :param activation: the activation function
        :param optimizer: the optimizer
        """
        self.input_size = input_size
        self.output_size = output_size
        self.density = density
        self.activation = activation
        self.optimizer = optimizer
        self.weight_initializer = weight_initializer
        self._layer = None

    def __str__(self):
        return f'Sparse(output_size={self.output_size}, density={self.density}, activation={self.activation}, optimizer={self.optimizer}, weight_initializer={self.weight_initializer})'

    def density_info(self) -> str:
        n, N = self._layer.W.nonzero_count()
        return f'{n}/{N} ({100 * n / N:.3f}%)'

    def compile(self, batch_size: int, dropout_rate: float=0.0):
        """
        Compiles the model into a C++ object

        :param batch_size: the batch size
        :param dropout_rate: the dropout rate
        :return:
        """
        activation = print_activation(self.activation)
        layer = nervalibrowwise.make_sparse_linear_layer(self.input_size, self.output_size, batch_size, self.density, activation, self.weight_initializer.compile(), self.optimizer.compile())
        self._layer = layer
        return layer

    def set_support_random(self, density: float) -> None:
        self._layer.set_support_random(density)

    def set_weights_and_bias(self, init: WeightInitializer) -> None:
        self._layer.set_weights_and_bias(init.compile())

    def initialize_weights(self, init: WeightInitializer) -> None:
        self._layer.initialize_weights(init.compile())

    def weight_count(self):
        return self._layer.weight_count()

    def positive_weight_count(self):
        return self._layer.positive_weight_count()

    def negative_weight_count(self):
        return self._layer.negative_weight_count()

    def prune_magnitude(self, zeta: float) -> int:
        return self._layer.prune_magnitude(zeta)

    def prune_SET(self, zeta: float) -> int:
        return self._layer.prune_SET(zeta)

    def prune_threshold(self, threshold: float) -> int:
        return self._layer.prune_threshold(threshold)

    def grow_random(self, count: int, weight_initializer=Xavier()) -> None:
        self._layer.grow_random(weight_initializer.compile(), count)


class BatchNormalization(Layer):
    def __init__(self,
                 input_size: int,
                 output_size: int
                ):
        assert input_size == output_size
        self.input_size = input_size
        self.output_size = output_size

    def compile(self, batch_size: int):
        return nervalibrowwise.batch_normalization_layer(self.input_size, batch_size)

    def __str__(self):
        return 'BatchNormalization()'


class SimpleBatchNormalization(Layer):
    def __init__(self,
                 input_size: int,
                 output_size: int
                ):
        assert input_size == output_size
        self.input_size = input_size
        self.output_size = output_size

    def compile(self, batch_size: int):
        return nervalibrowwise.simple_batch_normalization_layer(self.input_size, batch_size)

    def __str__(self):
        return 'SimpleBatchNormalization()'


class AffineTransform(Layer):
    def __init__(self,
                 input_size: int,
                 output_size: int
                ):
        assert input_size == output_size
        self.input_size = input_size
        self.output_size = output_size

    def compile(self, batch_size: int):
        return nervalibrowwise.affine_layer(self.input_size, batch_size)

    def __str__(self):
        return 'AffineTransform()'


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

    def compile(self, batch_size: int) -> None:
        self._check_layers()

        M = nervalibrowwise.MLP()

        # add layers
        for i, layer in enumerate(self.layers):
            cpp_layer = layer.compile(batch_size)
            M.append_layer(cpp_layer)
        self.compiled_model = M

    def feedforward(self, X):
        return self.compiled_model.feedforward(X)

    def backpropagate(self, Y, dY):
        return self.compiled_model.backpropagate(Y, dY)

    def optimize(self, eta):
        self.compiled_model.optimize(eta)

    def __str__(self):
        layers = ',\n  '.join([str(layer) for layer in self.layers])
        return f'Sequential(\n  {layers}\n)'

    def set_support_random(self):
        for layer in self.layers:
            if isinstance(layer, Sparse):
                layer.set_support_random(layer.density)

    def set_weights_and_bias(self, weight_initializers: List[WeightInitializer]):
        print(f'Initializing weights using {", ".join(str(w) for w in weight_initializers)}')
        self.compiled_model.set_weights_and_bias([w.compile() for w in weight_initializers])

    def load_weights_and_bias(self, filename: str):
        """
        Loads the weights and biases from a file in .npz format

        The weight matrices are stored using the keys W1, W2, ... and the bias vectors using the keys "b1, b2, ..."
        :param filename: the name of the file
        """
        print(f'Loading weights and bias from {filename}')
        self.compiled_model.load_weights_and_bias(filename)

    def save_weights_and_bias(self, filename: str):
        """
        Loads the weights and biases from a file in .npz format

        The weight matrices are stored using the keys W1, W2, ... and the bias vectors using the keys "b1, b2, ..."
        :param filename: the name of the file
        """
        print(f'Saving weights and bias to {filename}')
        self.compiled_model.save_weights_and_bias(filename)

    def info(self, msg):
        self.compiled_model.info(msg)


def compute_sparse_layer_densities(overall_density: float, layer_sizes: List[int], erk_power_scale: float=1) -> List[float]:
    return nervalibrowwise.compute_sparse_layer_densities(overall_density, layer_sizes, erk_power_scale)


def print_model_info(M: Sequential) -> None:
    """
    Prints detailed information about a multilayer perceptron
    :param M: a multilayer perceptron
    """
    nervalibrowwise.print_model_info(M.compiled_model)
