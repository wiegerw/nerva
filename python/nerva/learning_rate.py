# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from typing import List
import nervalib


class LearningRateScheduler(nervalib.learning_rate_scheduler):
    pass


class ConstantScheduler(nervalib.constant_scheduler):
    def __init__(self, lr: float):
        super().__init__(lr)

    def __str__(self):
        return f'ConstantScheduler(lr={self.lr})'


class TimeBasedScheduler(nervalib.time_based_scheduler):
    def __init__(self, lr: float, decay: float):
        super().__init__(lr, decay)

    def __str__(self):
        return f'TimeBasedScheduler(lr={self.lr}, decay={self.decay})'


class StepBasedScheduler(nervalib.step_based_scheduler):
    def __init__(self, lr: float, drop_rate: float, change_rate: float):
        super().__init__(lr, drop_rate, change_rate)

    def __str__(self):
        return f'StepBasedScheduler(lr={self.lr}, drop_rate={self.drop_rate}, change_rate={self.change_rate})'


class MultiStepLRScheduler(nervalib.multi_step_lr_scheduler):
    def __init__(self, lr: float, milestones: List[int], gamma: float):
        super().__init__(lr, milestones, gamma)

    def __str__(self):
        return f'MultiStepLRScheduler(lr={self.lr}, milestones={self.milestones}, gamma={self.gamma})'


class ExponentialScheduler(nervalib.exponential_scheduler):
    def __init__(self, lr: float, change_rate: float):
        super().__init__(lr, change_rate)

    def __str__(self):
        return f'ExponentialScheduler(lr={self.lr}, change_rate={self.change_rate})'
