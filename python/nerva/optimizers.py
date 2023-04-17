# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import re

class Optimizer(object):
    def compile(self):
        raise NotImplementedError


class GradientDescent(Optimizer):
    def compile(self):
        return 'GradientDescent'

    def __str__(self):
        return 'GradientDescent()'


class Momentum(Optimizer):
    def __init__(self, momentum: float):
        self.momentum = momentum

    def compile(self):
        return f'Momentum({self.momentum})'

    def __str__(self):
        return f'Momentum({self.momentum})'


class Nesterov(Optimizer):
    def __init__(self, momentum: float):
        self.momentum = momentum

    def compile(self):
        return f'Nesterov({self.momentum})'

    def __str__(self):
        return f'Nesterov({self.momentum})'


def parse_optimizer(text: str) -> Optimizer:
    if text == 'GradientDescent':
        return GradientDescent()
    elif text.startswith('Momentum'):
        m = re.match(r'Momentum\((.*)\)', text)
        momentum = float(m.group(1))
        return Momentum(momentum)
    elif text.startswith('Nesterov'):
        m = re.match(r'Nesterov\((.*)\)', text)
        momentum = float(m.group(1))
        return Momentum(momentum)
    raise RuntimeError(f"could not parse optimizer '{text}'")
