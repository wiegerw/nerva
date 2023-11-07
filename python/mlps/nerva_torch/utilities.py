# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import re
import time
from typing import Dict, Tuple, Union

import numpy as np
import torch


def set_numpy_options():
    np.set_printoptions(precision=8, edgeitems=3, threshold=5, suppress=True, linewidth=160)


def set_torch_options():
    torch.set_printoptions(precision=8, edgeitems=3, threshold=5, sci_mode=False, linewidth=160)

    # avoid 'Too many open files' error when using data loaders
    torch.multiprocessing.set_sharing_strategy('file_system')


def pp(name: str, x: torch.Tensor):
    if x.dim() == 1:
        print(f'{name} ({x.shape[0]})\n{x.data}')
    else:
        print(f'{name} ({x.shape[0]}x{x.shape[1]})\n{x.data}')


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
