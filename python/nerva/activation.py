# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)
from typing import Tuple

from nerva.utilities import parse_function_call


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


class TReLU(Activation):
    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def __str__(self):
        return f'TReLU({self.epsilon})'


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


class SReLU(Activation):
    def __init__(self, al: float = 0, tl: float = 0, ar: float = 0, tr: float = 1):
        self.al = al
        self.tl = tl
        self.ar = ar
        self.tr = tr

    def __str__(self):
        return f'SReLU(al={self.al},tl={self.tl},ar={self.ar},tr={self.tr})'


class HyperbolicTangent(Activation):
    def __str__(self):
        return 'HyperbolicTangent()'


def parse_activation(text: str) -> Tuple[Activation, float]:
    """
    Returns an activation function and a dropout rate
    :param text:
    :return:
    """
    func = parse_function_call(text)
    dropout = float(func.as_float('dropout')) if func.has_key('dropout') else 0.0
    if func.name == 'ReLU':
        return ReLU(), dropout
    elif func.name == 'Sigmoid':
        return Sigmoid(), dropout
    elif func.name == 'Softmax':
        return Softmax(), dropout
    elif func.name == 'LogSoftmax':
        return LogSoftmax(), dropout
    elif func.name == 'HyperbolicTangent':
        return HyperbolicTangent(), dropout
    elif func.name == 'TReLU':
        epsilon = func.as_float('epsilon')
        return TReLU(epsilon), dropout
    elif func.name == 'LeakyReLU':
        alpha = func.as_float('alpha')
        return LeakyReLU(alpha), dropout
    elif func.name == 'AllReLU':
        alpha = func.as_float('alpha')
        return AllReLU(alpha), dropout
    elif func.name == 'Linear':
        return NoActivation(), dropout
    elif func.name == 'SReLU':
        al = func.as_float('al', 0)
        tl = func.as_float('tl', 0)
        ar = func.as_float('ar', 0)
        tr = func.as_float('tr', 1)
        return SReLU(al, tl, ar, tr), dropout
    raise RuntimeError(f"could not parse activation '{text}'")
