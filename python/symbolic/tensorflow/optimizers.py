# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import os
from typing import List

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


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
        self.delta_x = tf.zeros_like(x)

    def update(self, eta):
        self.delta_x = self.mu * self.delta_x - eta * self.Dx
        self.x.assign(self.x + self.delta_x)


class NesterovOptimizer(MomentumOptimizer):
    def __init__(self, x, Dx, mu):
        super().__init__(x, Dx, mu)

    def update(self, eta):
        self.delta_x = self.mu * self.delta_x - eta * self.Dx
        self.x.assign(self.x + self.mu * self.delta_x - eta * self.Dx)
