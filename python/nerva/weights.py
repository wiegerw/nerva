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


class Zero(WeightInitializer):
    def compile(self):
        return nervalib.Weights.Zero

    def __str__(self):
        return 'Zero()'
