#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy, os
import re
import seaborn as sns
import matplotlib.cm as cm
import shutil
import os
import sys

import pathlib
import matplotlib.pyplot as plt


def parse_logfile(path: pathlib.Path) -> pd.DataFrame:
    text = path.read_text()
    lines = text.split("\n")
    pattern = re.compile(r'\s*epoch\s+(\d+)\s+lr:\s+(.*?)\sloss:\s+(.*?)\s+train accuracy:\s+(.*?)\s+test accuracy:\s+(.*?)\s+time:\s+(.*?)s?\s*$')

    epoch = []
    lr = []
    loss = []
    train_accuracy = []
    test_accuracy = []
    time = []

    for line in lines:
        m = pattern.search(line)
        if m:
            epoch.append(int(m.group(1)))
            lr.append(float(m.group(2)))
            loss.append(float(m.group(3)))
            train_accuracy.append(float(m.group(4)))
            test_accuracy.append(float(m.group(5)))
            time.append(float(m.group(6)))

    return pd.DataFrame({'epoch': epoch,
                         'lr': lr,
                         'loss': loss,
                         'train accuracy': train_accuracy,
                         'test accuracy': test_accuracy,
                         'time': time})


def make_plot(df_data, sparsity=None, labels=None, ax=None):

    palette = sns.color_palette()
    frameworks = {'nerva': {'label': 'Nerva ', 'color': palette[0]},
                  'torch': {'label': 'PyTorch ', 'color': palette[1]}}

    # for fw in frameworks:
    # ii = 0
    for fw in frameworks:
        data = df_data[df_data['framework'] == fw]
        for lb in labels:
            label = frameworks[fw]['label'] + lb
            if lb == 'train accuracy':
                ax = sns.lineplot(data=data[lb], label=label, color=frameworks[fw]['color'], linestyle='--')
            else:
                ax = sns.lineplot(data=data[lb], label=label, color=frameworks[fw]['color'])


    # ax.set_ylim([0.85, 1.01])
    # ax.set_ylim([0.0, 0.5])
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)

    sparsity_l = ['Sparsity: {}'.format(sparsity) if sparsity>0.0 else 'dense']

    if 'loss' in labels[0]:
        label_l = 'Loss'
        ax.set_ylim(bottom=0, top=3)
    else:
        label_l = 'Accuracy'
        ax.set_ylim(bottom=0, top=0.8)

    # ax.set_aspect(aspect=40)
    leg = ax.legend(fontsize=6)
    ax.grid(zorder=-10)
    plt.xlabel('Epoch')
    plt.ylabel('{}'.format(label_l))
    plt.title('{} vs Epoch ({})'.format(label_l, sparsity_l[0]))
    plt.subplots_adjust(hspace=0.8)
    plt.subplots_adjust(wspace=0.4)


    # Add a legend to the plot

def make_df_acc(df: pd.DataFrame):
    # make a new df with the best test accuracy for each framework and density and seed
    df_acc = df.drop(columns=['epoch', 'time', 'lr'])
    # df_acc = df_acc.groupby(['framework', 'seed']).sum().reset_index()

    return df_acc

def run(path_lc_logs, labels):
    density_l = ['dense', 0.5, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001]
    plt.figure(figsize=(8, 10))
    for i, d in enumerate(density_l):
        plt.subplot(4, 2, i+1)
        df = pd.DataFrame()

        d_label = ['sparse-{}'.format(d) if 'dense' not in str(d) else 'dense']
        for path in pathlib.Path(path_lc_logs).glob('*-{}-augmented-seed*.log'.format(d_label[0])):
            data = parse_logfile(path)

            framework = path.name.split('-')[0]
            seed = int(path.name.split('-')[-1].split('.')[0].replace('seed', ''))
            if 'dense' in path.name:
                density = 1.0
            else:
                density = float(path.name.split('-')[2])

            data['framework'] = framework
            data['density'] = density
            data['seed'] = seed

            df = pd.concat([df, data])

        df_acc = make_df_acc(df)
        make_plot(df_acc, sparsity=1-density, labels=labels, ax=None)

    label_l = ['Loss' if 'loss' in labels[0] else 'Accuracy']
    plt.savefig(f'{path_lc_logs}-{label_l[0]}.pdf')
    plt.close()


if __name__ == '__main__':
    folder = sys.argv[1]
    run(folder, labels=['loss'])
    run(folder, labels=['train accuracy', 'test accuracy'])

