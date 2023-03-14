# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

class Activation(object):
    pass


class NoActivation(Activation):
    def __str__(self):
        return 'NoActivation()'


class ReLU(Activation):
    def __str__(self):
        return 'ReLU()'


class Sigmoid(Activation):
    def __str__(self):
        return 'Sigmoid()'


class Softmax(Activation):
    def __str__(self):
        return 'Softmax()'


class LogSoftmax(Activation):
    def __str__(self):
        return 'LogSoftmax()'


class TrimmedReLU(Activation):
    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def __str__(self):
        return f'TrimmedReLU({self.epsilon})'


class LeakyReLU(Activation):
    def __init__(self, alpha: float):
        self.alpha = alpha

    def __str__(self):
        return f'LeakyReLU({self.alpha})'


class AllReLU(Activation):
    def __init__(self, alpha: float):
        self.alpha = alpha

    def __str__(self):
        return f'AllReLU({self.alpha})'


class HyperbolicTangent(Activation):
    def __str__(self):
        return 'HyperbolicTangent()'
