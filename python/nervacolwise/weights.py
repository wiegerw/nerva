# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

class WeightInitializer(object):
    pass


class Xavier(WeightInitializer):
    def __str__(self):
        return 'Xavier'


class XavierNormalized(WeightInitializer):
    def __str__(self):
        return 'XavierNormalized'


class He(WeightInitializer):
    def __str__(self):
        return 'He'


class Uniform(WeightInitializer):
    def __str__(self):
        return 'Uniform'


class PyTorch(WeightInitializer):
    def __str__(self):
        return 'PyTorch'


class Zero(WeightInitializer):
    def __str__(self):
        return 'Zero'


class None_(WeightInitializer):
    def __str__(self):
        return 'None'


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
