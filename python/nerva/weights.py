# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import nervalib

class WeightInitializer(object):
    def compile(self):
        return NotImplementedError


class Xavier(WeightInitializer):
    def compile(self):
        return 'Xavier'

    def __str__(self):
        return 'Xavier()'

class XavierNormalized(WeightInitializer):
    def compile(self):
        return 'XavierNormalized'

    def __str__(self):
        return 'XavierNormalized()'


class He(WeightInitializer):
    def compile(self):
        return 'He'

    def __str__(self):
        return 'He()'


class Uniform(WeightInitializer):
    def compile(self):
        return 'Uniform'

    def __str__(self):
        return 'Uniform()'


class PyTorch(WeightInitializer):
    def compile(self):
        return 'PyTorch'

    def __str__(self):
        return 'PyTorch()'


class Zero(WeightInitializer):
    def compile(self):
        return 'Zero'

    def __str__(self):
        return 'Zero()'


class None_(WeightInitializer):
    def compile(self):
        return 'None'

    def __str__(self):
        return 'None_()'


def parse_weight_initializer(text: str):
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
    elif text == 'None':
        return None_()
    raise RuntimeError(f"unknown weight initializer '{text}'")
