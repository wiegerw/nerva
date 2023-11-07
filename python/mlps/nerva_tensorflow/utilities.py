# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import os
import re
import time
from typing import Dict, Tuple

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


def parse_function_call(text: str) -> Tuple[str, Dict[str, str]]:
    text = text.strip()
    try:
        if re.match(r"\w*$", text):  # no arguments case
            name = text
            return name, {}
        else:
            m = re.match(r"(\w*)\((.*?)\)", text)
            name = m.group(1)
            args = {}
            for arg in m.group(2).split(','):
                key, value = arg.split('=')
                key = key.strip()
                value = value.strip()
                if key in args:
                    raise ValueError(f'Duplicate key in function call "{text}"')
                args[key] = value
            return name, args
    except Exception as e:
        print(e)
        pass
    raise RuntimeError(f'Could not parse function call "{text}"')


def load_dict_from_npz(filename: str) -> Dict[str, np.ndarray]:
    """
    Loads a dictionary from a file in .npz format
    :param filename: a file name
    :return: a dictionary
    """

    return dict(np.load(filename, allow_pickle=True))
