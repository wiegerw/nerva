# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import numpy as np


def to_one_hot(x: np.ndarray, n_classes: int):
    one_hot = np.zeros((len(x), n_classes))
    one_hot[np.arange(len(x)), x] = 1
    return one_hot
