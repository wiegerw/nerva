# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf


def set_numpy_options():
    np.set_printoptions(precision=8, edgeitems=3, threshold=5, suppress=True, linewidth=160)


def set_tensorflow_options():
    # N.B. Tensorflow doesn't support print options
    import numpy as np
    np.set_printoptions(precision=8, edgeitems=3, threshold=5, suppress=False, linewidth=160)


def pp(name: str, x: tf.Tensor):
    shape = tf.shape(x).numpy()
    if tf.rank(x) == 1:
        print(f'{name} ({shape[0]})\n{x.numpy()}')
    else:
        print(f'{name} ({shape[0]}x{shape[1]})\n{x.numpy()}')


class StopWatch(object):
    def __init__(self):
        self.start = time.perf_counter()

    def seconds(self):
        end = time.perf_counter()
        return end - self.start

    def reset(self):
        self.start = time.perf_counter()
