import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy, os
import re
import seaborn as sns
import matplotlib.cm as cm
import shutil
import os

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


def make_plot(df_data, sparsity=None, labels=None):

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
                sns.lineplot(data=data[lb], label=label, color=frameworks[fw]['color'],  linestyle='--')
            else:
                sns.lineplot(data=data[lb], label=label, color=frameworks[fw]['color'])
            # ii += 1


    # ax.set_ylim([0.85, 1.01])
    # ax.set_ylim([0.0, 0.5])
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)

    label_l = ['Loss' if 'loss' in labels[0] else 'Accuracy']

    plt.xlabel('Epoch')
    plt.ylabel('{}'.format(label_l[0]))
    plt.title('{} vs Epoch (Sparsity: {})'.format(label_l[0], sparsity))

    # Add a legend to the plot
    # plt.savefig('./pics/three-nerva-runs/{}.png'.format(str(1)))
    plt.savefig('./seed12_plots/{}.png'.format(str(1)))
    plt.close()

def make_df_acc(df: pd.DataFrame):
    # make a new df with the best test accuracy for each framework and density and seed
    df_acc = df.drop(columns=['epoch', 'time', 'lr'])
    # df_acc = df_acc.groupby(['framework', 'seed']).sum().reset_index()

    return df_acc

def main():

    # path_lc_logs = './pics/three-nerva-runs'
    path_lc_logs = './seed12'
    labels = ['train accuracy', 'test accuracy']
    # labels = ['loss']

    df = pd.DataFrame()
    for path in pathlib.Path(path_lc_logs).glob('*-sparse-0.1-augmented-seed*.log'):
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
    make_plot(df_acc, sparsity=1-density, labels=labels)


if __name__ == '__main__':

    main()
