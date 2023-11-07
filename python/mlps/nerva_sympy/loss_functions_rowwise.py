# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import sympy as sp

from mlps.nerva_sympy.loss_functions import Cross_entropy_loss_rowwise, Cross_entropy_loss_rowwise_gradient, \
    Logistic_cross_entropy_loss_rowwise, Logistic_cross_entropy_loss_rowwise_gradient, Mean_squared_error_loss_rowwise, \
    Mean_squared_error_loss_rowwise_gradient, Negative_log_likelihood_loss_rowwise, \
    Negative_log_likelihood_loss_rowwise_gradient, Softmax_cross_entropy_loss_rowwise, \
    Softmax_cross_entropy_loss_rowwise_gradient, Squared_error_loss_rowwise, Squared_error_loss_rowwise_gradient, \
    Stable_softmax_cross_entropy_loss_rowwise, Stable_softmax_cross_entropy_loss_rowwise_gradient

Matrix = sp.Matrix


class LossFunction(object):
    def __call__(self, Y: Matrix, T: Matrix) -> float:
        raise NotImplementedError

    def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
        raise NotImplementedError


class SquaredErrorLossFunction(LossFunction):
    def __call__(self, Y: Matrix, T: Matrix) -> float:
        return Squared_error_loss_rowwise(Y, T)

    def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
        return Squared_error_loss_rowwise_gradient(Y, T)


class MeanSquaredErrorLossFunction(LossFunction):
    def __call__(self, Y: Matrix, T: Matrix) -> float:
        return Mean_squared_error_loss_rowwise(Y, T)

    def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
        return Mean_squared_error_loss_rowwise_gradient(Y, T)


class CrossEntropyLossFunction(LossFunction):
    def __call__(self, Y: Matrix, T: Matrix) -> float:
        return Cross_entropy_loss_rowwise(Y, T)

    def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
        return Cross_entropy_loss_rowwise_gradient(Y, T)


class SoftmaxCrossEntropyLossFunction(LossFunction):
    def __call__(self, Y: Matrix, T: Matrix) -> float:
        return Softmax_cross_entropy_loss_rowwise(Y, T)

    def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
        return Softmax_cross_entropy_loss_rowwise_gradient(Y, T)


class StableSoftmaxCrossEntropyLossFunction(LossFunction):
    def __call__(self, Y: Matrix, T: Matrix) -> float:
        return Stable_softmax_cross_entropy_loss_rowwise(Y, T)

    def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
        return Stable_softmax_cross_entropy_loss_rowwise_gradient(Y, T)


class LogisticCrossEntropyLossFunction(LossFunction):
    def __call__(self, Y: Matrix, T: Matrix) -> float:
        return Logistic_cross_entropy_loss_rowwise(Y, T)

    def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
        return Logistic_cross_entropy_loss_rowwise_gradient(Y, T)


class NegativeLogLikelihoodLossFunction(LossFunction):
    def __call__(self, Y: Matrix, T: Matrix) -> float:
        return Negative_log_likelihood_loss_rowwise(Y, T)

    def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
        return Negative_log_likelihood_loss_rowwise_gradient(Y, T)
