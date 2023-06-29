from typing import List

from symbolic.sympy.layers_rowwise import BatchNormalizationLayer, Layer, LinearLayer, SigmoidLayer, SoftmaxLayer, \
    LogSoftmaxLayer, SReLULayer, ActivationLayer
from symbolic.sympy.loss_functions_rowwise import LossFunction, SquaredErrorLossFunction, MeanSquaredErrorLossFunction, \
    CrossEntropyLossFunction, StableSoftmaxCrossEntropyLossFunction, LogisticCrossEntropyLossFunction, \
    NegativeLogLikelihoodLossFunction
from symbolic.sympy.multilayer_perceptron_rowwise import MultilayerPerceptron
from symbolic.sympy.parse_mlp import parse_optimizer, parse_activation


def parse_multilayer_perceptron(layer_specifications: List[str],
                                linear_layer_sizes: List[int],
                                optimizers: List[str],
                                linear_layer_weight_initializers: List[str],
                                batch_size: int
                               ) -> MultilayerPerceptron:

    assert len(linear_layer_weight_initializers) == len(linear_layer_sizes) - 1
    layers = []

    linear_layer_index = 0
    optimizer_index = 0
    D = linear_layer_sizes[linear_layer_index]  # the input size of the current layer
    N = batch_size

    for specification in layer_specifications:
        if specification == 'BatchNormalization':
            layer = BatchNormalizationLayer(D, N)
            optimizer = optimizers[optimizer_index]
            layer.set_optimizer(parse_optimizer(optimizer))
            optimizer_index += 1
        else:
            K = linear_layer_sizes[linear_layer_index + 1]  # the output size of the layer
            optimizer = optimizers[optimizer_index]
            weight_initializer = linear_layer_weight_initializers[linear_layer_index]
            layer = parse_linear_layer(specification, D, K, N, optimizer, weight_initializer)
            optimizer_index += 1
            linear_layer_index += 1
            D = K
        layers.append(layer)
    return MultilayerPerceptron(layers)


def parse_linear_layer(text: str,
                       D: int,
                       K: int,
                       N: int,
                       optimizer: str,
                       weight_initializer: str
                      ) -> Layer:
    if text == 'Linear':
        layer = LinearLayer(D, K, N)
    elif text == 'Sigmoid':
        layer = SigmoidLayer(D, K, N)
    elif text == 'Softmax':
        layer = SoftmaxLayer(D, K, N)
    elif text == 'LogSoftmax':
        layer = LogSoftmaxLayer(D, K, N)
    elif text.startswith('SReLU'):
        act = parse_activation(text)
        layer = SReLULayer(D, K, N, act)
    else:
        act = parse_activation(text)
        layer = ActivationLayer(D, K, N, act)
    layer.optimizer = parse_optimizer(optimizer, layer)
    layer.set_weights(weight_initializer)
    return layer


def parse_loss_function(text: str) -> LossFunction:
    if text == "SquaredError":
        return SquaredErrorLossFunction()
    elif text == "MeanSquaredError":
        return MeanSquaredErrorLossFunction()
    elif text == "CrossEntropy":
        return CrossEntropyLossFunction()
    elif text == "SoftmaxCrossEntropy":
        return StableSoftmaxCrossEntropyLossFunction()
    elif text == "LogisticCrossEntropy":
        return LogisticCrossEntropyLossFunction()
    elif text == "NegativeLogLikelihood":
        return NegativeLogLikelihoodLossFunction()
    else:
        raise RuntimeError(f"unknown loss function '{text}'")
