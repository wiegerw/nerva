# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import os
import re
import time
from typing import Union, Dict, Tuple
import jax.numpy as jnp
import numpy as np
import sympy as sp
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import torch


def to_numpy(x: Union[sp.Matrix, np.ndarray, torch.Tensor, tf.Tensor, tf.Variable, jnp.ndarray]) -> np.ndarray:
    if isinstance(x, sp.Matrix):
        return np.array(x.tolist(), dtype=np.float64)
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, tf.Tensor):
        return x.numpy()
    elif isinstance(x, tf.Variable):
        return x.numpy()
    elif isinstance(x, jnp.ndarray):
        return np.array(x)
    else:
        raise ValueError("Unsupported input type. Input must be one of sp.Matrix, np.ndarray, torch.Tensor, or tf.Tensor.")


def to_sympy(X: np.ndarray) -> sp.Matrix:
    return sp.Matrix(X)


def to_torch(X: np.ndarray) -> torch.Tensor:
    return torch.Tensor(X)


def to_tensorflow(X: np.ndarray) -> tf.Tensor:
    return tf.convert_to_tensor(X)


def to_jax(X: np.ndarray) -> jnp.ndarray:
    return jnp.array(X)


def to_eigen(X: np.ndarray) -> np.ndarray:
    return np.asfortranarray(np.copy(X, order='C'))


class StopWatch(object):
    def __init__(self):
        self.start = time.perf_counter()

    def seconds(self):
        end = time.perf_counter()
        return end - self.start

    def reset(self):
        self.start = time.perf_counter()


def contains_any_char(text: str, chars: str):
    """ Check whether string text contains any character in chars."""
    return 1 in [c in text for c in chars]


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


def load_dict_from_npz(filename: str) -> Dict[str, Union[torch.Tensor, torch.LongTensor]]:
    """
    Loads a dictionary from a file in .npz format
    :param filename: a file name
    :return: a dictionary
    """
    def make_tensor(x: np.ndarray) -> Union[torch.Tensor, torch.LongTensor]:
        if np.issubdtype(x.dtype, np.integer):
            return torch.LongTensor(x)
        return torch.Tensor(x)

    data = dict(np.load(filename, allow_pickle=True))
    data = {key: make_tensor(value) for key, value in data.items()}
    return data
