# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

class Activation(object):
    pass


class NoActivation(Activation):
    pass


class ReLU(Activation):
    pass


class Sigmoid(Activation):
    pass


class Softmax(Activation):
    pass


class LeakyReLU(Activation):
    def __init__(self, alpha: float):
        self.alpha = alpha


class AllReLU(Activation):
    def __init__(self, alpha: float):
        self.alpha = alpha


class HyperbolicTangent(Activation):
    pass
