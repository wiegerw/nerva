# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import tensorflow as tf

from mlps.nerva_tensorflow.loss_functions import Cross_entropy_loss_colwise, Cross_entropy_loss_colwise_gradient, \
    Logistic_cross_entropy_loss_colwise, Logistic_cross_entropy_loss_colwise_gradient, Mean_squared_error_loss_colwise, \
    Mean_squared_error_loss_colwise_gradient, Negative_log_likelihood_loss_colwise, \
    Negative_log_likelihood_loss_colwise_gradient, Softmax_cross_entropy_loss_colwise, \
    Softmax_cross_entropy_loss_colwise_gradient, Squared_error_loss_colwise, Squared_error_loss_colwise_gradient, \
    Stable_softmax_cross_entropy_loss_colwise, Stable_softmax_cross_entropy_loss_colwise_gradient

Matrix = tf.Tensor


class LossFunction(object):
    def __call__(self, Y: Matrix, T: Matrix) -> float:
        raise NotImplementedError

    def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
        raise NotImplementedError


class SquaredErrorLossFunction(LossFunction):
    def __call__(self, Y: Matrix, T: Matrix) -> float:
        return Squared_error_loss_colwise(Y, T)

    def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
        return Squared_error_loss_colwise_gradient(Y, T)


class MeanSquaredErrorLossFunction(LossFunction):
    def __call__(self, Y: Matrix, T: Matrix) -> float:
        return Mean_squared_error_loss_colwise(Y, T)

    def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
        return Mean_squared_error_loss_colwise_gradient(Y, T)


class CrossEntropyLossFunction(LossFunction):
    def __call__(self, Y: Matrix, T: Matrix) -> float:
        return Cross_entropy_loss_colwise(Y, T)

    def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
        return Cross_entropy_loss_colwise_gradient(Y, T)


class SoftmaxCrossEntropyLossFunction(LossFunction):
    def __call__(self, Y: Matrix, T: Matrix) -> float:
        return Softmax_cross_entropy_loss_colwise(Y, T)

    def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
        return Softmax_cross_entropy_loss_colwise_gradient(Y, T)


class StableSoftmaxCrossEntropyLossFunction(LossFunction):
    def __call__(self, Y: Matrix, T: Matrix) -> float:
        return Stable_softmax_cross_entropy_loss_colwise(Y, T)

    def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
        return Stable_softmax_cross_entropy_loss_colwise_gradient(Y, T)


class LogisticCrossEntropyLossFunction(LossFunction):
    def __call__(self, Y: Matrix, T: Matrix) -> float:
        return Logistic_cross_entropy_loss_colwise(Y, T)

    def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
        return Logistic_cross_entropy_loss_colwise_gradient(Y, T)


class NegativeLogLikelihoodLossFunction(LossFunction):
    def __call__(self, Y: Matrix, T: Matrix) -> float:
        return Negative_log_likelihood_loss_colwise(Y, T)

    def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
        return Negative_log_likelihood_loss_colwise_gradient(Y, T)


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
