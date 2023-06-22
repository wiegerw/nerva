# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import re
from symbolic.torch.matrix_operations import zeros
from symbolic.optimizers import Optimizer, CompositeOptimizer, GradientDescentOptimizer


class MomentumOptimizer(GradientDescentOptimizer):
    def __init__(self, x, Dx, mu):
        super().__init__(x, Dx)
        self.mu = mu

        m, n = x.shape
        self.delta_x = zeros(m, n)

    def update(self, eta):
        self.delta_x = self.mu * self.delta_x - eta * self.Dx
        self.x += self.delta_x


class NesterovOptimizer(GradientDescentOptimizer):
    def __init__(self, x, Dx, mu):
        super().__init__(x, Dx)
        self.mu = mu

        m, n = x.shape
        self.delta_x = zeros(m, n)
        self.delta_x_prev = zeros(m, n)

    def update(self, eta):
        self.delta_x_prev = self.delta_x
        self.delta_x = self.mu * self.delta_x - eta * self.Dx
        self.x += (-self.mu * self.delta_x_prev + (1 + self.mu) * self.delta_x)


def parse_optimizer(text: str,
                    layer
                   ) -> Optimizer:
    try:
        if text == 'GradientDescent':
            return CompositeOptimizer([GradientDescentOptimizer(layer.W, layer.DW), GradientDescentOptimizer(layer.b, layer.Db)])
        elif text.startswith('Momentum'):
            m = re.match(r'Momentum\((.*)\)$', text)
            mu = float(m.group(1))
            return CompositeOptimizer([MomentumOptimizer(layer.W, layer.DW, mu), MomentumOptimizer(layer.b, layer.Db, mu)])
        elif text.startswith('Nesterov'):
            m = re.match(r'Nesterov\((.*)\)$', text)
            mu = float(m.group(1))
            return CompositeOptimizer([NesterovOptimizer(layer.W, layer.DW, mu), NesterovOptimizer(layer.b, layer.Db, mu)])
    except:
        pass
    raise RuntimeError(f'Could not parse optimizer "{text}"')
