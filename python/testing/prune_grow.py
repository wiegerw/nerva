#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import re
from typing import List
from nerva.weights import WeightInitializer, parse_weight_initializer
from nerva.layers import Sparse, Sequential


def parse_arguments(text: str, name: str, n: int) -> List[str]:
    pattern = name + r'\((.*?)\)'
    m = re.match(pattern, text)
    if not m:
        return []
    result = list(filter(None, m.group(1).split(',')))
    if len(result) != n:
        return []
    return result


class RegrowFunction(object):
    """
    Interface for pruning + growing the sparse layers of a neural network
    """
    def __call__(self, M: Sequential):
        raise NotImplementedError


class PruneFunction(object):
    """
    Interface for pruning the weights of a sparse layer
    """
    def __call__(self, layer: Sparse):
        raise NotImplementedError


class PruneMagnitude(PruneFunction):
    """
    Prunes a fraction zeta of the weights in the sparse layers. Weights
    are pruned according to their magnitude.
    """
    def __init__(self, zeta):
        self.zeta = zeta

    def __call__(self, layer: Sparse):
        return layer.prune_magnitude(self.zeta)


class PruneThreshold(PruneFunction):
    """
    Prunes weights with magnitude below a given threshold.
    """
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, layer: Sparse):
        return layer.prune_threshold(self.threshold)


class PruneSET(PruneFunction):
    """
    Prunes a fraction zeta of the positive and a fraction zeta of the negative
    weights in the sparse layers. Weights are pruned according to their magnitude.
    """
    def __init__(self, zeta):
        self.zeta = zeta

    def __call__(self, layer: Sparse):
        return layer.prune_SET(self.zeta)


def parse_prune_strategy(strategy: str):
    arguments = parse_arguments(strategy, 'Magnitude', 1)
    if arguments:
        return PruneMagnitude(float(arguments[0]))

    arguments = parse_arguments(strategy, 'Threshold', 1)
    if arguments:
        return PruneThreshold(float(arguments[0]))

    arguments = parse_arguments(strategy, 'SET', 1)
    if arguments:
        return PruneSET(float(arguments[0]))

    raise RuntimeError(f"unknown prune strategy '{strategy}'")


class GrowStrategy(object):
    def __call__(self, layer: Sparse, count: int):
        raise NotImplementedError


class GrowRandom(GrowStrategy):
    def __init__(self, init: WeightInitializer):
        self.init = init

    def __call__(self, layer: Sparse, count: int):
        layer.grow_random(count, self.init)


def parse_grow_strategy(strategy: str, init: WeightInitializer):
    if strategy == 'Random':
        return GrowRandom(init)
    else:
        raise RuntimeError(f"unknown grow strategy '{strategy}'")


class PruneGrow(RegrowFunction):
    def __init__(self, prune_strategy: str, grow_strategy: str, weights: str):
        init = parse_weight_initializer(weights)
        self.prune = parse_prune_strategy(prune_strategy)
        self.grow = parse_grow_strategy(grow_strategy, init)

    def __call__(self, M: Sequential):
        for layer in M.layers:
            if isinstance(layer, Sparse):
                weight_count = layer.weight_count()
                count = self.prune(layer)
                print(f'pruning + growing {count}/{weight_count} weights')
                self.grow(layer, count)
