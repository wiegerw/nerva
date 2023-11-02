# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import time

import numpy as np


def set_numpy_options():
    np.set_printoptions(precision=8, edgeitems=3, threshold=5, suppress=True, linewidth=160)


def pp(name: str, x: np.ndarray):
    if x.ndim == 1:
        print(f'{name} ({x.shape[0]})\n{x}')
    else:
        print(f'{name} ({x.shape[0]}x{x.shape[1]})\n{x}')


class StopWatch(object):
    def __init__(self):
        self.start = time.perf_counter()

    def seconds(self):
        end = time.perf_counter()
        return end - self.start

    def reset(self):
        self.start = time.perf_counter()
