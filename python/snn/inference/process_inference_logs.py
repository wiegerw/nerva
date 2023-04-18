#!/usr/bin/env python3

import pathlib
import re
import sys
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# inference-batch-size-100-density-0.001-seed-1.log:Average PyTorch inference time for density=0.001 batch_size=100: 9.7779ms
def parse_results(filename: str) -> pd.DataFrame:
    text = pathlib.Path(filename).read_text()
    lines = list(filter(None, text.split('\n')))
    pattern = re.compile(r'inference-batch-size-(\d+)-density-([\d.]+)-seed-(\d+).log:Average (\w+) inference.*:\s+([\d.]+)ms')

    batch_size = []
    density = []
    seed = []
    framework = []
    time = []

    for line in lines:
        m = pattern.search(line)
        if m:
            batch_size.append(int(m.group(1)))
            density.append(float(m.group(2)))
            seed.append(int(m.group(3)))
            framework.append(m.group(4))
            time.append(float(m.group(5)))

    return pd.DataFrame({'framework': framework,
                         'density': density,
                         'seed': seed,
                         'batch_size': batch_size,
                         'time': time})


def set_log_scale(x_axis: str):
    # Make the xaxis log scale from 0 to 1, such that very small densities like 0.001 and 0.005 are distinguishable
    if x_axis == 'density':
        plt.xscale('log')
        # plt.xlim(xmax=1)
        plt.xticks([0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1],
                   ['0.001', '0.005', '0.01', '0.05', '0.1', '0.2', '0.5', '1'])
    else:
        plt.xscale('logit')
        plt.xticks([0.5, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999],
                   ['0.5', '0.8', '0.9', '0.95', '0.99', '0.995', '0.999'])


def make_time_vs_density_plot(df: pd.DataFrame, paths: List[pathlib.Path]):
    palette = sns.color_palette()
    color_nerva = palette[0]
    color_torch = palette[1]

    def make_sparse_df(framework):
        d = df[df.framework == framework]
        d = d.drop(d[d.sparsity == 0.0].index)  # drop the sparsity == 0.0 rows
        return d

    # makes a straight line from the sparsity==0.0 measurement
    def make_dense_df(framework):
        d = df.copy(deep=True)
        d = d[d.framework == framework]
        d = pd.concat([d, d], ignore_index=True)
        d.loc[:int(len(d)/2)-1, 'sparsity'] = 0.5
        d.loc[int(len(d)/2):, 'sparsity'] = 0.999
        return d

    data = make_sparse_df('Nerva')
    sns.lineplot(x='sparsity', y='time', data=data, marker='o', label='Nerva', color=color_nerva)

    data = make_sparse_df('PyTorch')
    sns.lineplot(x='sparsity', y='time', data=data, marker='o', label='PyTorch', color=color_torch)

    data = make_dense_df('Nerva')
    sns.lineplot(x='sparsity', y='time', data=data, linestyle='--', label='Nerva dense', color=color_nerva)

    data = make_dense_df('PyTorch')
    sns.lineplot(x='sparsity', y='time', data=data, linestyle='--', label='PyTorch dense', color=color_torch)

    x_axis = 'sparsity'
    plt.grid(zorder=0)
    plt.ylim(ymin=0)
    set_log_scale(x_axis)

    # Add labels and title to the plot
    xlabel = x_axis[0].upper() + x_axis[1:]  # make first letter x_axis uppercase
    plt.xlabel(xlabel)
    plt.ylabel('Inference time (ms)')
    plt.title(f'Inference time vs Sparsity')

    # Add a legend to the plot. Make sure nerva is presented as Nerva and torch as PyTorch
    plt.legend()

    # Save the plot
    for path in paths:
        print(f'Saving plot to {path}')
        plt.savefig(path, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    result_file = sys.argv[1]
    df = parse_results(result_file)

    # drop the batch size 100 results
    df = df[df.batch_size == 1]
    df = df.drop(columns=['batch_size'])

    # replace density by sparsity
    df['sparsity'] = 1 - df.density
    df = df.drop(columns=['density'])

    # sort by sparsity
    df = df.sort_values(by=['framework', 'sparsity'])

    pdf_file = pathlib.Path('./seed123-inference-vs-sparsity.pdf')
    svg_file = pdf_file.with_suffix('.svg')
    make_time_vs_density_plot(df, [pdf_file, svg_file])

