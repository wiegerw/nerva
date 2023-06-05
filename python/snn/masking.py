#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import random
from typing import List

import torch


def reservoir_sample(k: int, n: int) -> List[int]:
    # Initialize the reservoir with the first k elements
    reservoir = [i for i in range(k)]

    # Iterate over the remaining elements
    for i in range(k, n):
        j = random.randint(0, i)
        if j < k:
            reservoir[j] = i

    return reservoir


def create_mask(W: torch.Tensor, non_zero_count: int) -> torch.Tensor:
    """
    Creates a boolean matrix with the same shape as W, and with exactly non_zero_count positions equal to 1
    :param W:
    :param non_zero_count:
    :return:
    """
    mask = torch.zeros_like(W)

    I = reservoir_sample(non_zero_count, W.numel())  # generates non_zero_count random indices in W
    for i in I:
        row = i // W.size(1)
        col = i % W.size(1)
        mask[row, col] = 1

    return mask
