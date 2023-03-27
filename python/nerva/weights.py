# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import nervalib

class WeightInitializer(object):
    def compile(self):
        return NotImplementedError


class Xavier(WeightInitializer):
    def compile(self):
        return nervalib.Weights.Xavier

    def __str__(self):
        return 'Xavier()'

class XavierNormalized(WeightInitializer):
    def compile(self):
        return nervalib.Weights.XavierNormalized

    def __str__(self):
        return 'XavierNormalized()'


class He(WeightInitializer):
    def compile(self):
        return nervalib.Weights.He

    def __str__(self):
        return 'He()'


class Uniform(WeightInitializer):
    def compile(self):
        return nervalib.Weights.Uniform

    def __str__(self):
        return 'Uniform()'


class PyTorch(WeightInitializer):
    def compile(self):
        return nervalib.Weights.PyTorch

    def __str__(self):
        return 'PyTorch()'


class Zero(WeightInitializer):
    def compile(self):
        return nervalib.Weights.Zero

    def __str__(self):
        return 'Zero()'


def parse_weights(text: str):
    if text == 'Xavier':
        return Xavier()
    elif text == 'XavierNormalized':
        return XavierNormalized()
    elif text == 'He':
        return He()
    elif text == 'Uniform':
        return Uniform()
    elif text == 'Zero':
        return Zero()
    elif text == 'PyTorch':
        return PyTorch()
    raise RuntimeError(f"unknown weight initializer '{text}'")
