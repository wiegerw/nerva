# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

class Optimizer(object):
    def compile(self):
        raise NotImplementedError


class GradientDescent(Optimizer):
    def compile(self):
        return 'gradient-descent'


class Momentum(Optimizer):
    def __init__(self, momentum: float):
        self.momentum = momentum

    def compile(self):
        return f'momentum({self.momentum})'


class Nesterov(Optimizer):
    def __init__(self, momentum: float):
        self.momentum = momentum

    def compile(self):
        return f'nesterov({self.momentum})'
