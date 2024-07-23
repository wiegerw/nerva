# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from nervacolwise.utilities import parse_function_call


class Optimizer(object):
    def compile(self):
        raise NotImplementedError


class GradientDescent(Optimizer):
    def __str__(self):
        return 'GradientDescent()'


class Momentum(Optimizer):
    def __init__(self, momentum: float):
        self.momentum = momentum

    def __str__(self):
        return f'Momentum({self.momentum})'


class Nesterov(Optimizer):
    def __init__(self, momentum: float):
        self.momentum = momentum

    def __str__(self):
        return f'Nesterov({self.momentum})'


def parse_optimizer(text: str) -> Optimizer:
    func = parse_function_call(text)
    if func.name =='GradientDescent':
        return GradientDescent()
    elif func.name =='Momentum':
        momentum = func.as_float('momentum')
        return Momentum(momentum)
    elif func.name =='Nesterov':
        momentum = func.as_float('momentum')
        return Nesterov(momentum)
    raise RuntimeError(f"could not parse optimizer '{text}'")
