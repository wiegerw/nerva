# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from typing import Any, Callable, List

from mlps.nerva_numpy.optimizers import GradientDescentOptimizer, MomentumOptimizer, NesterovOptimizer
from mlps.nerva_sympy.matrix_operations import zeros


class Optimizer(object):
    def update(self, eta):
        raise NotImplementedError


class CompositeOptimizer(Optimizer):
    def __init__(self, optimizers: List[Optimizer]):
        self.optimizers = optimizers

    def update(self, eta):
        for optimizer in self.optimizers:
            optimizer.update(eta)


class GradientDescentOptimizer(Optimizer):
    def __init__(self, x, Dx):
        self.x = x
        self.Dx = Dx

    def update(self, eta):
        self.x -= eta * self.Dx


class MomentumOptimizer(GradientDescentOptimizer):
    def __init__(self, x, Dx, mu):
        super().__init__(x, Dx)
        self.mu = mu

        m, n = x.shape
        self.delta_x = zeros(m, n)

    def update(self, eta):
        self.delta_x = self.mu * self.delta_x - eta * self.Dx
        self.x += self.delta_x


class NesterovOptimizer(MomentumOptimizer):
    def __init__(self, x, Dx, mu):
        super().__init__(x, Dx, mu)

    def update(self, eta):
        self.delta_x = self.mu * self.delta_x - eta * self.Dx
        self.x += self.mu * self.delta_x - eta * self.Dx


def parse_optimizer(text: str) -> Callable[[Any, Any], Optimizer]:
    try:
        name, args = parse_function_call(text)
        if name == 'GradientDescent':
            return lambda x, Dx: GradientDescentOptimizer(x, Dx)
        elif name == 'Momentum':
            mu = float(args['mu'])
            return lambda x, Dx: MomentumOptimizer(x, Dx, mu)
        elif name == 'Nesterov':
            mu = float(args['mu'])
            return lambda x, Dx: NesterovOptimizer(x, Dx, mu)
    except:
        pass
    raise RuntimeError(f'Could not parse optimizer "{text}"')
