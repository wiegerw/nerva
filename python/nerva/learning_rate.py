# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import re
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


def parse_learning_rate(text: str) -> LearningRateScheduler:
    if text.startswith('Constant'):
        m = re.match(r'Constant\((.*)\)', text)
        lr = float(m.group(1))
        return ConstantScheduler(lr)
    elif text.startswith('TimeBased'):
        m = re.match(r'TimeBased\((.*),(.*)\)', text)
        lr = float(m.group(1))
        decay = float(m.group(2))
        return TimeBasedScheduler(lr, decay)
    elif text.startswith('StepBased'):
        m = re.match(r'StepBased\((.*),(.*),(.*)\)', text)
        lr = float(m.group(1))
        drop_rate = float(m.group(2))
        change_rate = float(m.group(3))
        return StepBasedScheduler(lr, drop_rate, change_rate)
    elif text.startswith('MultiStepLR'):
        m = re.match(r'MultiStepLR\((.*);(.*);(.*)\)', text)
        lr = float(m.group(1))
        milestones = [int(x) for x in m.group(2).split(',')]
        gamma = float(m.group(3))
        return MultiStepLRScheduler(lr, milestones, gamma)
    elif text.startswith('Exponential'):
        m = re.match(r'Exponential\((.*),(.*)\)', text)
        lr = float(m.group(1))
        change_rate = float(m.group(2))
        return ExponentialScheduler(lr, change_rate)
    raise RuntimeError(f"could not parse learning rate scheduler '{text}'")
