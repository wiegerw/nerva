# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import nervalib

class WeightInitializer(object):
    def compile(self):
        raise NotImplementedError


class Xavier(WeightInitializer):
    def compile(self):
        return nervalib.Weights.Xavier


class KaiMing(WeightInitializer):
    def compile(self):
        return nervalib.Weights.He
