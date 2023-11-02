# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from typing import List

from symbolic.sympy.matrix_operations import zeros


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
