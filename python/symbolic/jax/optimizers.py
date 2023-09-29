# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from typing import List

import jax.numpy as jnp

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
    def __init__(self, obj, attr_x: str, attr_Dx: str):
        """
        Store the names of the x and Dx attributes
        """
        self.obj = obj
        self.attr_x = attr_x
        self.attr_Dx = attr_Dx

    def update(self, eta):
        x = getattr(self.obj, self.attr_x)
        Dx = getattr(self.obj, self.attr_Dx)
        x1 = x - eta * Dx
        setattr(self.obj, self.attr_x, x1)


class MomentumOptimizer(GradientDescentOptimizer):
    def __init__(self, obj, attr_x: str, attr_Dx: str, mu: float):
        super().__init__(obj, attr_x, attr_Dx)
        self.mu = mu
        x = getattr(self.obj, self.attr_x)
        self.delta_x = jnp.zeros_like(x)

    def update(self, eta):
        x = getattr(self.obj, self.attr_x)
        Dx = getattr(self.obj, self.attr_Dx)
        self.delta_x = self.mu * self.delta_x - eta * Dx
        x1 = x + self.delta_x
        setattr(self.obj, self.attr_x, x1)

class NesterovOptimizer(MomentumOptimizer):
    def __init__(self, obj, attr_x: str, attr_Dx: str, mu: float):
        super().__init__(obj, attr_x, attr_Dx, mu)

    def update(self, eta):
        x = getattr(self.obj, self.attr_x)
        Dx = getattr(self.obj, self.attr_Dx)
        self.delta_x_prev = self.delta_x
        self.delta_x = self.mu * self.delta_x - eta * Dx
        x1 = x + self.mu * self.delta_x - eta * Dx
        setattr(self.obj, self.attr_x, x1)
