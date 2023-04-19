# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import re

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
    def __init__(self, al: float, tl: float, ar: float, tr: float):
        self.al = al
        self.tl = tl
        self.ar = ar
        self.tr = tr

    def __str__(self):
        return f'SReLU({self.al},{self.tl},{self.ar},{self.tr})'


class HyperbolicTangent(Activation):
    def __str__(self):
        return 'HyperbolicTangent()'


def parse_activation(text: str) -> Activation:
    if text == 'ReLU':
        return ReLU()
    elif text == 'Sigmoid':
        return Sigmoid()
    elif text == 'Softmax':
        return Softmax()
    elif text == 'LogSoftmax':
        return LogSoftmax()
    elif text == 'HyperbolicTangent':
        return HyperbolicTangent()
    elif text.startswith('TReLU'):
        m = re.match(r'TReLU\((.*)\)', text)
        alpha = float(m.group(1))
        return TReLU(alpha)
    elif text.startswith('LeakyReLU'):
        m = re.match(r'LeakyReLU\((.*)\)', text)
        alpha = float(m.group(1))
        return LeakyReLU(alpha)
    elif text.startswith('AllReLU'):
        m = re.match(r'AllReLU\((.*)\)', text)
        alpha = float(m.group(1))
        return AllReLU(alpha)
    elif text == 'Linear':
        return NoActivation()
    elif text.startswith('SReLU'):
        m = re.match(r'SReLU\((.*),(.*),(.*),(.*)\)', text)
        al = float(m.group(1))
        tl = float(m.group(2))
        ar = float(m.group(3))
        tr = float(m.group(4))
        return SReLU(al, tl, ar, tr)
    raise RuntimeError(f"could not parse activation '{text}'")
