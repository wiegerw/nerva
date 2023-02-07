#!/usr/bin/env python3

import math
import pathlib
import re
from collections import defaultdict
from typing import Union

import pandas as pd
import matplotlib.pyplot as plt

# N.B. This scripts assumes there are logs for 3 seeds

def round_(x: Union[float, str]) -> str:
    if isinstance(x, str):
        return x
    return f'{x:.2f}'


def create_width_table(model, width, size, time):
    n = len(model)

    table = defaultdict(lambda: {'nervatime': 0.0, 'torchtime': 0.0, 'factor': '-'})
    for i in range(n):
        if size[i] == 1024:
            item = table[width[i]]
            if model[i] == 'nerva':
                item['nervatime'] += time[i]
            elif model[i] == 'torch':
                item['torchtime'] += time[i]

    width_ = []
    nervatime = []
    torchtime = []
    factor = []

    for w in sorted(table.keys()):
        item = table[w]
        if isinstance(item['nervatime'], float) and isinstance(item['torchtime'], float):
            item['factor'] = item['torchtime'] / item['nervatime']
        width_.append(w)
        nervatime.append(round_(item['nervatime'] / 3))
        torchtime.append(round_(item['torchtime'] / 3))
        factor.append(round_(item['factor']))

    print(width_)
    print(nervatime)
    print(torchtime)
    print(factor)

    df = pd.DataFrame({'$N': width_,
                       'Nerva': nervatime,
                       'PyTorch': torchtime,
                       'factor': factor})
    return df

def create_size_table(model, width, size, time):
    n = len(model)
    table = defaultdict(lambda: {'nervatime': 0.0, 'torchtime': 0.0, 'factor': '-'})

    for i in range(n):
        if width[i] == 3:
            item = table[size[i]]
            if model[i] == 'nerva':
                item['nervatime'] += time[i]
            elif model[i] == 'torch':
                item['torchtime'] += time[i]

    size_ = []
    nervatime = []
    torchtime = []
    factor = []

    for s in sorted(table.keys()):
        item = table[s]
        if isinstance(item['nervatime'], float) and isinstance(item['torchtime'], float):
            item['factor'] = item['torchtime'] / item['nervatime']
        size_.append(s)
        nervatime.append(round_(item['nervatime'] / 3))
        torchtime.append(round_(item['torchtime'] / 3))
        factor.append(round_(item['factor']))

    df = pd.DataFrame({'size': size_,
                       'Nerva': nervatime,
                       'PyTorch': torchtime,
                       'factor': factor}
                      )
    print(size_)
    return df


# torch-density-0.01-sizes-2048,2048,2048-seed-12345.log:epoch   1  lr: 0.10000000  loss: 2.30671000  train accuracy: 0.10000000  test accuracy: 0.10000000  time: 55.98063940s
def parse_results() -> None:
    text = pathlib.Path('results').read_text()
    lines = list(filter(None, text.split('\n')))
    pattern = re.compile(r'(\w+)-density-([\d.]+)-sizes-([\d,]+)-seed-(\d+).log.*time:\s* ([\d.]*)s\s*$')

    model = []
    density = []
    width = []  # the number layers
    size = []  # the size of the layers
    time = []
    seed = []

    for line in lines:
        m = pattern.search(line)
        if m:
            model.append(m.group(1))
            density.append(float(m.group(2)))
            architecture = m.group(3)
            width.append(architecture.count(',') + 1)
            size.append(int(architecture.split(',')[0]))
            seed.append(int(m.group(4)))
            time.append(float(m.group(5)))

    df = pd.DataFrame({'model': model,
                       'density': density,
                       'width': width,
                       'size': size,
                       'time': time})
    # print(df)

    df1 = create_width_table(model, width, size, time)
    print(df1.to_latex(index=False))

    df2 = create_size_table(model, width, size, time)
    print(df2.to_latex(index=False))


if __name__ == '__main__':
    parse_results()
