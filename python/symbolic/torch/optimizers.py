# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import re
from collections.abc import Callable
from typing import Any

import torch
from symbolic.optimizers import Optimizer, GradientDescentOptimizer


class MomentumOptimizer(GradientDescentOptimizer):
    def __init__(self, x, Dx, mu):
        super().__init__(x, Dx)
        self.mu = mu
        self.delta_x = torch.zeros_like(x)

    def update(self, eta):
        self.delta_x = self.mu * self.delta_x - eta * self.Dx
        self.x += self.delta_x


class NesterovOptimizer(GradientDescentOptimizer):
    def __init__(self, x, Dx, mu):
        super().__init__(x, Dx)
        self.mu = mu
        self.delta_x = torch.zeros_like(x)
        self.delta_x_prev = torch.zeros_like(x)

    def update(self, eta):
        self.delta_x_prev = self.delta_x
        self.delta_x = self.mu * self.delta_x - eta * self.Dx
        self.x += (-self.mu * self.delta_x_prev + (1 + self.mu) * self.delta_x)


def parse_optimizer(text: str) -> Callable[[Any, Any], Optimizer]:
    try:
        if text == 'GradientDescent':
            return lambda x, Dx: GradientDescentOptimizer(x, Dx)
        elif text.startswith('Momentum'):
            m = re.match(r'Momentum\((.*)\)$', text)
            mu = float(m.group(1))
            return lambda x, Dx: MomentumOptimizer(x, Dx, mu)
        elif text.startswith('Nesterov'):
            m = re.match(r'Nesterov\((.*)\)$', text)
            mu = float(m.group(1))
            return lambda x, Dx: NesterovOptimizer(x, Dx, mu)
    except:
        pass
    raise RuntimeError(f'Could not parse optimizer "{text}"')
