# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from typing import Optional, List, Union, Tuple
from nerva.activation import Activation, NoActivation, ReLU, AllReLU, Softmax, LogSoftmax, Sigmoid, HyperbolicTangent, \
    LeakyReLU, TrimmedReLU
from nerva.optimizers import Optimizer, GradientDescent
from nerva.weights import WeightInitializer, Xavier
import nervalib


class Layer(object):
    pass


class Dense(Layer):
    def __init__(self,
                 units: int,
                 activation: Activation=NoActivation(),
                 optimizer: Optimizer=GradientDescent(),
                 weight_initializer: WeightInitializer=Xavier()
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
        self._layer = None

    def __str__(self):
        return f'Dense(units={self.units}, activation={self.activation}, optimizer={self.optimizer}, weight_initializer={self.weight_initializer})'

    def density_info(self) -> str:
        N = self._layer.W.size
        return f'{N}/{N} (100%)'

    def compile(self, batch_size: int, dropout_rate: float=0.0):
        """
        Compiles the model into a C++ object

        :param batch_size: the batch size
        :param dropout_rate: the dropout rate
        :return:
        """
        layer = None
        if dropout_rate == 0.0:
            if isinstance(self.activation, NoActivation):
                layer = nervalib.linear_layer(self.input_size, self.units, batch_size)
            elif isinstance(self.activation, AllReLU):
                layer = nervalib.all_relu_layer(self.activation.alpha, self.input_size, self.units, batch_size)
            elif isinstance(self.activation, HyperbolicTangent):
                layer = nervalib.hyperbolic_tangent_layer(self.input_size, self.units, batch_size)
            elif isinstance(self.activation, LeakyReLU):
                layer = nervalib.leaky_relu_layer(self.activation.alpha, self.input_size, self.units, batch_size)
            elif isinstance(self.activation, LogSoftmax):
                layer = nervalib.log_softmax_layer(self.input_size, self.units, batch_size)
            elif isinstance(self.activation, ReLU):
                layer = nervalib.relu_layer(self.input_size, self.units, batch_size)
            elif isinstance(self.activation, Sigmoid):
                layer = nervalib.sigmoid_layer(self.input_size, self.units, batch_size)
            elif isinstance(self.activation, Softmax):
                layer = nervalib.softmax_layer(self.input_size, self.units, batch_size)
            elif isinstance(self.activation, TrimmedReLU):
                layer = nervalib.trimmed_relu_layer(self.activation.epsilon, self.input_size, self.units, batch_size)
        else:
            if isinstance(self.activation, NoActivation):
                layer = nervalib.linear_dropout_layer(self.input_size, self.units, batch_size, dropout_rate)
            elif isinstance(self.activation, AllReLU):
                layer = nervalib.all_relu_dropout_layer(self.activation.alpha, self.input_size, self.units, batch_size, dropout_rate)
            elif isinstance(self.activation, HyperbolicTangent):
                layer = nervalib.hyperbolic_tangent_dropout_layer(self.input_size, self.units, batch_size, dropout_rate)
            elif isinstance(self.activation, LeakyReLU):
                layer = nervalib.leaky_relu_dropout_layer(self.activation.alpha, self.input_size, self.units, batch_size, dropout_rate)
            elif isinstance(self.activation, ReLU):
                layer = nervalib.relu_dropout_layer(self.input_size, self.units, batch_size, dropout_rate)
            elif isinstance(self.activation, Sigmoid):
                layer = nervalib.sigmoid_dropout_layer(self.input_size, self.units, batch_size, dropout_rate)
            elif isinstance(self.activation, Softmax):
                layer = nervalib.softmax_dropout_layer(self.input_size, self.units, batch_size, dropout_rate)

        if not layer:
            raise RuntimeError('Unsupported layer type')

        layer.initialize_weights(self.weight_initializer.compile())
        self._layer = layer
        return layer


class Sparse(Layer):
    def __init__(self, units: int, density: float, activation: Activation=NoActivation(), optimizer=GradientDescent(), weight_initializer=Xavier()):
        """
        A sparse layer.

        :param units: the number of outputs of the layer
        :param density: the density of the layer. This is a number between 0.0 (fully sparse) and 1.0 (fully dense).
        :param activation: the activation function
        :param optimizer: the optimizer
        """
        self.units = int(units) if not isinstance(units, int) else units
        if self.units < 0:
            raise ValueError(
                f"Received an invalid value for `units`, expected "
                f"a positive integer. Received: units={units}"
            )
        self.density = density
        self.activation = activation
        self.optimizer = optimizer
        self.weight_initializer = weight_initializer
        self.input_size = -1
        self._layer = None

    def __str__(self):
        return f'Sparse(units={self.units}, density={self.density}, activation={self.activation}, optimizer={self.optimizer}, weight_initializer={self.weight_initializer})'

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
        layer = None
        if dropout_rate == 0.0:
            if isinstance(self.activation, NoActivation):
                layer = nervalib.sparse_linear_layer(self.input_size, self.units, batch_size, self.density)
            elif isinstance(self.activation, AllReLU):
                layer = nervalib.sparse_all_relu_layer(self.activation.alpha, self.input_size, self.units, batch_size, self.density)
            elif isinstance(self.activation, HyperbolicTangent):
                layer = nervalib.sparse_hyperbolic_tangent_layer(self.input_size, self.units, batch_size, self.density)
            elif isinstance(self.activation, LogSoftmax):
                layer = nervalib.sparse_log_softmax_layer(self.input_size, self.units, batch_size, self.density)
            elif isinstance(self.activation, LeakyReLU):
                layer = nervalib.sparse_leaky_relu_layer(self.activation.alpha, self.input_size, self.units, batch_size, self.density)
            elif isinstance(self.activation, ReLU):
                layer = nervalib.sparse_relu_layer(self.input_size, self.units, batch_size, self.density)
            elif isinstance(self.activation, Sigmoid):
                layer = nervalib.sparse_sigmoid_layer(self.input_size, self.units, batch_size, self.density)
            elif isinstance(self.activation, Softmax):
                layer = nervalib.sparse_softmax_layer(self.input_size, self.units, batch_size, self.density)
            elif isinstance(self.activation, TrimmedReLU):
                layer = nervalib.sparse_trimmed_relu_layer(self.activation.epsilon, self.input_size, self.units, batch_size, self.density)

        if not layer:
            raise RuntimeError('Unsupported layer type')

        layer.initialize_weights(self.weight_initializer.compile())
        self._layer = layer
        return layer

    def weight_count(self):
        return self._layer.weight_count()

    def positive_weight_count(self):
        return self._layer.positive_weight_count()

    def negative_weight_count(self):
        return self._layer.negative_weight_count()

    def prune_magnitude(self, zeta: float):
        return self._layer.prune_magnitude(zeta)

    def prune_SET(self, zeta: float):
        return self._layer.prune_SET(zeta)

    def prune_threshold(self, threshold: float):
        return self._layer.prune_threshold(threshold)

    def grow_random(self, count: int, weight_initializer=Xavier()):
        self._layer.grow_random(weight_initializer.compile(), count)


class Dropout(Layer):
    def __init__(self, rate: float):
        self.rate = rate

    def __str__(self):
        return f'Dropout({self.rate})'


class BatchNormalization(Layer):
    def __init__(self):
        self.input_size = -1

    def compile(self, batch_size: int):
        return nervalib.batch_normalization_layer(self.input_size, batch_size)

    def __str__(self):
        return 'BatchNormalization()'


class SimpleBatchNormalization(Layer):
    def __init__(self):
        self.input_size = -1

    def compile(self, batch_size: int):
        return nervalib.simple_batch_normalization_layer(self.input_size, batch_size)

    def __str__(self):
        return 'SimpleBatchNormalization()'


class AffineTransform(Layer):
    def __init__(self):
        self.input_size = -1

    def compile(self, batch_size: int):
        return nervalib.affine_layer(self.input_size, batch_size)

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

        # Dropout may only appear after Dense
        for i, layer in enumerate(layers):
            if isinstance(layer, Dropout):
                if i == 0:
                    raise RuntimeError('The first layer cannot be a dropout layer')
                if not isinstance(layers[i-1], Dense):
                    raise RuntimeError(f'Dropout layer {i} is not preceded by a Dense layer')

    def compile(self, input_size: int, batch_size: int) -> None:
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
                cpp_layer = layer.compile(batch_size, dropout_rate)
                cpp_layer.set_optimizer(layer.optimizer.compile())
                cpp_layer.initialize_weights(layer.weight_initializer.compile())
                M.append_layer(cpp_layer)
            elif isinstance(layer, (BatchNormalization, SimpleBatchNormalization, AffineTransform)):
                layer.input_size = input_size
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

    # def info(self):
    #     self.compiled_model.info('M')

    def load_weights(self, filename: str):
        self.compiled_model.import_weights_npz(filename)

    def save_weights(self, filename: str):
        self.compiled_model.export_weights_npz(filename)


def compute_sparse_layer_densities(overall_density: float, layer_sizes: List[int], erk_power_scale: float=1) -> List[float]:
    return nervalib.compute_sparse_layer_densities(overall_density, layer_sizes, erk_power_scale)
