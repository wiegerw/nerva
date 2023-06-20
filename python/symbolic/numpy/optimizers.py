# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import re
from symbolic.numpy.matrix_operations import *


class Optimizer(object):
    def update(self, eta):
        raise NotImplementedError


class GradientDescentOptimizer(Optimizer):
    def __init__(self, W, DW, b, Db):
        self.W = W
        self.DW = DW
        self.b = b
        self.Db = Db

    def update(self, eta):
        self.W -= eta * self.DW
        self.b -= eta * self.Db


class MomentumOptimizer(GradientDescentOptimizer):
    def __init__(self, W, DW, b, Db, mu):
        super().__init__(W, DW, b, Db)
        self.mu = mu

        m, n = W.shape
        self.delta_W = zeros(m, n)

        m, n = b.shape
        self.delta_b = zeros(m, n)

    def update(self, eta):
        self.delta_W = self.mu * self.delta_W - eta * self.DW
        self.W += self.delta_W
        self.delta_b = self.mu * self.delta_b - eta * self.Db
        self.b += self.delta_b


class NesterovOptimizer(GradientDescentOptimizer):
    def __init__(self, W, DW, b, Db, mu):
        super().__init__(W, DW, b, Db)
        self.mu = mu

        m, n = W.shape
        self.delta_W = zeros(m, n)
        self.delta_W_prev = zeros(m, n)

        m, n = b.shape
        self.delta_b = zeros(m, n)
        self.delta_b_prev = zeros(m, n)

    def update(self, eta):
        self.delta_W_prev = self.delta_W
        self.delta_W = self.mu * self.delta_W - eta * self.DW
        self.W += (-self.mu * self.delta_W_prev + (1 + self.mu) * self.delta_W)
        self.delta_b_prev = self.delta_b
        self.delta_b = self.mu * self.delta_b - eta * self.Db
        self.b += (-self.mu * self.delta_b_prev + (1 + self.mu) * self.delta_b)


def parse_optimizer(text: str,
                    layer
                   ) -> Optimizer:
    try:
        if text == 'GradientDescent':
            return GradientDescentOptimizer(layer.W, layer.DW, layer.b, layer.Db)
        elif text.startswith('Momentum'):
            m = re.match(r'Momentum\((.*)\)$', text)
            mu = float(m.group(1))
            return MomentumOptimizer(layer.W, layer.DW, layer.b, layer.Db, mu)
        elif text.startswith('Nesterov'):
            m = re.match(r'Nesterov\((.*)\)$', text)
            mu = float(m.group(1))
            return NesterovOptimizer(layer.W, layer.DW, layer.b, layer.Db, mu)
    except:
        pass
    raise RuntimeError(f'Could not parse optimizer "{text}"')
