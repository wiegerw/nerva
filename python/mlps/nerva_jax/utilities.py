# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import re
import time
from typing import Dict

import jax.numpy as jnp


def set_jax_options():
    jnp.set_printoptions(precision=8, edgeitems=3, threshold=5, suppress=True, linewidth=160)


def pp(name: str, x: jnp.ndarray):
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


class FunctionCall:
    def __init__(self, name: str, arguments: dict):
        self.name = name
        self.arguments = arguments

    def has_key(self, key: str) -> bool:
        return key in self.arguments

    def get_value(self, key: str) -> str:
        if key in self.arguments:
            return self.arguments[key]
        elif len(self.arguments) == 1 and "" in self.arguments:
            return self.arguments[""]
        return ""

    def as_scalar(self, key: str, default_value: float = None) -> float:
        value = self.get_value(key)
        if value:
            return float(value)  # Assuming scalar is a float, adjust if necessary
        elif default_value:
            return default_value
        raise RuntimeError(f"Could not find an argument named \"{key}\"")

    def as_string(self, key: str, default_value: str = "") -> str:
        value = self.get_value(key)
        if value:
            return value
        elif default_value:
            return default_value
        raise RuntimeError(f"Could not find an argument named \"{key}\"")


def parse_function_call(text: str) -> FunctionCall:
    """
    Parse a string of the shape `NAME(key1=value1, key2=value2, ...)`.
    If there are no arguments the parentheses may be omitted.
    If there is only one parameter, it is allowed to pass `NAME(value)` instead of `NAME(key=value)`
    """

    text = text.strip()

    def error():
        raise RuntimeError(f"Could not parse function call \"{text}\"")

    name = ""
    arguments = {}

    # no parentheses
    match = re.match(r"(\w+$)", text)
    if match:
        name = match.group(1)
        return FunctionCall(name, arguments)

    # with parentheses
    match = re.match(r"(\w+)\((.*?)\)", text)
    if match:
        name = match.group(1)
        args = re.split(r",", match.group(2))

        if len(args) == 1 and "=" not in args[0]:
            # NAME(value)
            value = args[0].strip()
            arguments[""] = value
            return FunctionCall(name, arguments)
        else:
            # NAME(key1=value1, ...)
            for arg in args:
                words = re.split(r"\s*=\s*", arg.strip())
                if len(words) != 2:
                    error()
                key, value = words
                if key in arguments:
                    print(f"Key \"{key}\" appears multiple times.")
                    error()
                arguments[key] = value
            return FunctionCall(name, arguments)

    error()


def load_dict_from_npz(filename: str) -> Dict[str, jnp.ndarray]:
    """
    Loads a dictionary from a file in .npz format
    :param filename: a file name
    :return: a dictionary
    """

    return dict(jnp.load(filename, allow_pickle=True))
