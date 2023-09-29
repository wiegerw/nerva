# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from symbolic.optimizers import GradientDescentOptimizer


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
