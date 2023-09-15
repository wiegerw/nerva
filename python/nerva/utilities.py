# Copyright 2022 - 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import re
from typing import Union, Dict

import numpy as np
import torch
import time


class StopWatch(object):
    def __init__(self):
        self.start = time.perf_counter()

    def seconds(self):
        end = time.perf_counter()
        return end - self.start

    def reset(self):
        self.start = time.perf_counter()


class MapTimer:
    """
    Timer class with a map interface. For each key a start and stop value is stored.
    """

    def __init__(self):
        self.values = {}

    def start(self, key):
        self.values[key] = (time.perf_counter(), None)

    def stop(self, key):
        t = time.perf_counter()
        self.values[key] = (self.values[key][0], t)

    def milliseconds(self, key):
        t1, t2 = self.values[key]
        return (t2 - t1) * 1000

    def seconds(self, key):
        t1, t2 = self.values[key]
        return t2 - t1


def flatten_numpy(x: np.ndarray) -> np.ndarray:
    shape = x.shape
    return x.reshape(shape[0], -1)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return np.asfortranarray(x.detach().numpy().T)


def to_one_hot_numpy(x: np.ndarray, n_classes: int):
    return flatten_numpy(np.asfortranarray(np.eye(n_classes)[x].T))


def torch_inf_norm(x: torch.Tensor) -> float:
    """
    Returns the infinity norm of a tensor
    :param x: a tensor
    """
    return torch.abs(x).max().item()


def pp(name: str, x: Union[torch.Tensor, np.ndarray]):
    if isinstance(x, np.ndarray):
        x = torch.Tensor(x)
    if x.dim() == 1:
        print(f'{name} ({x.shape[0]}) norm = {torch_inf_norm(x):.8f}\n{x.data}')
    else:
        print(f'{name} ({x.shape[0]}x{x.shape[1]}) norm = {torch_inf_norm(x):.8f}\n{x.data}')


class FunctionCall(object):
    def __init__(self, name: str, arguments: Dict[str, str]):
        self.name = name
        self.arguments = arguments

    def has_key(self, key: str) -> bool:
        return key in self.arguments

    def get_value(self, key: str):
        if key in self.arguments:
            return self.arguments[key]
        if len(self.arguments) == 1 and '' in self.arguments:
            return self.arguments['']
        return None

    def as_float(self, key: str, default_value: float = None) -> float:
        """
        Returns the argument named key as a float
        :param key: the name of the argument
        :param default_value: a default value that is returned if no value was found
        """
        value = self.get_value(key)
        if value:
            return float(value)
        if default_value is not None:
            return default_value
        raise RuntimeError(f'could not find an argument named "{key}"')

    def as_string(self, key: str, default_value: str = None) -> str:
        value = self.get_value(key)
        if value:
            return value
        if default_value is not None:
            return default_value
        raise RuntimeError(f'Could not find an argument named "{key}"')


def parse_function_call(text: str) -> FunctionCall:
    text = text.strip()
    try:
        # no parentheses
        if re.match(r"\w*$", text):  # no arguments case
            name = text
            return FunctionCall(name, {})

        # with parentheses
        m = re.match(r"(\w*)\((.*?)\)", text)
        name = m.group(1)
        arguments = {}
        args = m.group(2).split(',')

        if len(args) == 1 and '=' not in args[0]:
            # NAME(value)
            value = args[0].strip()
            arguments[''] = value
            return FunctionCall(name, arguments)
        else:
            # NAME(key1=value1, ...)
            for arg in args:
                words = re.split(r'\s*=\s*', arg)
                if len(words) != 2:
                    raise RuntimeError(f'Could not parse function call "{text}"')
                key, value = words
                if key in arguments:
                    print(f'Key "{key}" appears multiple times.')
                    raise RuntimeError(f'Could not parse function call "{text}"')
                arguments[key] = value
        return FunctionCall(name, arguments)
    except Exception as e:
        print(e)
        pass
    raise RuntimeError(f'Could not parse function call "{text}"')
