# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import re
from typing import List

import sympy as sp


class LearningRateScheduler(object):
    def __call__(self, epoch: int):
        raise NotImplementedError


class ConstantScheduler(LearningRateScheduler):
    def __init__(self, lr: float):
        self.lr = lr

    def __str__(self):
        return f'ConstantScheduler(lr={self.lr})'

    def __call__(self, epoch: int):
        return self.lr


class TimeBasedScheduler(LearningRateScheduler):
    def __init__(self, lr: float, decay: float):
        self.lr = lr
        self.decay = decay

    def __str__(self):
        return f'TimeBasedScheduler(lr={self.lr}, decay={self.decay})'

    def __call__(self, epoch: int) -> float:
        self.lr = self.lr / (1 + self.decay * float(epoch))
        return self.lr


class StepBasedScheduler(LearningRateScheduler):
    def __init__(self, lr: float, drop_rate: float, change_rate: float):
        self.lr = lr
        self.drop_rate = drop_rate
        self.change_rate = change_rate

    def __str__(self):
        return f'StepBasedScheduler(lr={self.lr}, drop_rate={self.drop_rate}, change_rate={self.change_rate})'

    def __call__(self, epoch: int) -> float:
        return self.lr * sp.Pow(self.drop_rate, sp.floor((1.0 + epoch) / self.change_rate))


class MultiStepLRScheduler(LearningRateScheduler):
    def __init__(self, lr: float, milestones: List[int], gamma: float):
        self.lr = lr
        self.milestones = milestones
        self.gamma = gamma

    def __str__(self):
        return f'MultiStepLRScheduler(lr={self.lr}, milestones={self.milestones}, gamma={self.gamma})'

    def __call__(self, epoch: int) -> float:
        eta = self.lr
        for milestone in self.milestones:
            if epoch >= milestone:
                eta *= self.gamma
            else:
                break
        return eta


class ExponentialScheduler(LearningRateScheduler):
    def __init__(self, lr: float, change_rate: float):
        self.lr = lr
        self.change_rate = change_rate

    def __str__(self):
        return f'ExponentialScheduler(lr={self.lr}, change_rate={self.change_rate})'

    def __call__(self, epoch: int) -> float:
        return self.lr * sp.exp(-self.change_rate * float(epoch))


def parse_learning_rate(text: str) -> LearningRateScheduler:
    try:
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
    except:
        pass
    raise RuntimeError(f"could not parse learning rate scheduler '{text}'")
