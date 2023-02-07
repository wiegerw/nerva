#!/usr/bin/env python3

import math
import pathlib
import re
from collections import defaultdict
from typing import Union

import pandas as pd
import matplotlib.pyplot as plt


def make_plot(df: pd.DataFrame, path: pathlib.Path):
    df_nerva_100 = df[(df['model'] == 'Nerva') & (df['batch_size'] == 100)]
    df_nerva_1 = df[(df['model'] == 'Nerva') & (df['batch_size'] == 1)]
    df_pytorch_100 = df[(df['model'] == 'PyTorch') & (df['batch_size'] == 100)]
    df_pytorch_1 = df[(df['model'] == 'PyTorch') & (df['batch_size'] == 1)]

    plt.plot(df_nerva_1['density'], df_nerva_1['time'], label='nerva batch-size 1')
    plt.plot(df_nerva_100['density'], df_nerva_100['time'], label='nerva batch-size 100')
    plt.plot(df_pytorch_1['density'], df_pytorch_1['time'], label='pytorch batch-size 1')
    plt.plot(df_pytorch_100['density'], df_pytorch_100['time'], label='pytorch batch-size 100')

    # Add labels and title to the plot
    plt.xlabel('Density')
    plt.ylabel('Time')
    plt.title(f'Density vs Time ({path})')

    # Add a legend to the plot
    plt.legend()

    # Show the plot
    plt.show()

    # Save the plot
    plt.savefig(path.with_suffix('.png'))

    plt.close()


# inference-batch-size-100-density-0.1.log:Average Nerva inference time for density=0.1 batch_size=100: 6.0605ms
def parse_results() -> pd.DataFrame:
    text = pathlib.Path('results').read_text()
    lines = list(filter(None, text.split('\n')))
    pattern = re.compile(r'inference-batch-size-(\d+)-density-([\d.]+).log:Average (\w+) inference.*:\s+([\d.]+)ms')

    batch_size = []
    density = []
    model = []
    time = []

    for line in lines:
        m = pattern.search(line)
        if m:
            batch_size.append(int(m.group(1)))
            density.append(float(m.group(2)))
            model.append(m.group(3))
            time.append(float(m.group(4)))

    return pd.DataFrame({'model': model,
                         'density': density,
                         'batch_size': batch_size,
                         'time': time})


if __name__ == '__main__':
    df = parse_results()
    make_plot(df, pathlib.Path('inference.png'))
