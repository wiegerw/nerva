#!/usr/bin/env python3

import pathlib
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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

def make_acc_vs_density_plot(df: pd.DataFrame, path: pathlib.Path, x_axis: str = 'density', log_scale: bool = False):
    # plot 'accuracy' vs 'density' for each framework using seaborn
    sns.lineplot(x=x_axis, y='test accuracy', hue='framework', data=df, marker='o')

    # Make the y-axis between 0 and 1
    plt.ylim(0, 1)

    if log_scale:
        if x_axis == 'density':
            # Make the xaxis log scale from 0 to 1, such that very small numbers like 0.001 and 0.005 are distinguishable
            plt.xscale('log')
            plt.xlim(xmax=1)
            # Make the x-axis ticks on the specific density levels
            plt.xticks([ 0.001,   0.005,   0.01,   0.05,   0.1,   0.2,   0.5,   1],
                       ['0.001', '0.005', '0.01', '0.05', '0.1', '0.2', '0.5', '1'])
        else:
            # Make the xaxis log scale from 0 to 1, such that numbers close to 1 like 0.999 and 0.995 are distinguishable
            plt.xscale('logit')
            # plt.xlim(xmax=1)
            # Make the x-axis ticks on the specific sparsity levels
            plt.xticks([ 0.5,   0.8,   0.9,   0.95,   0.99,   0.995,   0.999],
                       ['0.5', '0.8', '0.9', '0.95', '0.99', '0.995', '0.999'])
            # TODO: dense run should be plotted as a horizontal line (because sparsity=0 does not work in logit scale)

    # Add labels and title to the plot
    xlabel = x_axis[0].upper() + x_axis[1:]  # make first letter x_axis uppercase
    plt.xlabel(xlabel)
    plt.ylabel('Best test accuracy')
    plt.title(f'Accuracy vs {xlabel}')
    # Add a legend to the plot
    plt.legend()
    # Save the plot
    plt.savefig(path)
    plt.close()


def make_time_vs_density_plot(df: pd.DataFrame, path: pathlib.Path, x_axis: str = 'density'):
    # plot 'time' vs 'density/sparsity' for each framework using seaborn
    sns.lineplot(x=x_axis, y='time', hue='framework', data=df, marker='o')

    # Add labels and title to the plot
    xlabel = x_axis[0].upper() + x_axis[1:]  # make first letter x_axis uppercase
    plt.xlabel(xlabel)
    plt.ylabel('Training time (minutes)')
    plt.title(f'Time vs {xlabel}')
    # Add a legend to the plot
    plt.legend()
    # Save the plot
    plt.savefig(path)
    plt.close()


def main():
    df = pd.DataFrame()
    for path in pathlib.Path('./logs').glob('*.log'):
        data = parse_logfile(path)

        # example path.name = 'nerva-sparse-0.1-seed1.log'
        framework = path.name.split('-')[0]
        density = float(path.name.split('-')[2])
        seed = int(path.name.split('-')[-1].split('.')[0].replace('seed', ''))

        data['framework'] = framework
        data['density'] = density
        data['seed'] = seed

        df = pd.concat([df, data])

    # make a new df with the total time summed for each framework and density and seed
    df_time = df.drop(columns=['epoch', 'lr', 'loss', 'train accuracy', 'test accuracy'])
    df_time = df_time.groupby(['framework', 'density', 'seed']).sum().reset_index()
    df_time['time'] = df_time['time'] / 60  # convert time to minutes
    df_time['sparsity'] = 1 - df_time['density']

    # make a new df with the best test accuracy for each framework and density and seed
    df_best = df.drop(columns=['epoch', 'time', 'loss'])
    df_best = df_best.groupby(['framework', 'density', 'seed']).max().reset_index()
    df_best['sparsity'] = 1 - df_best['density']


    x_axis = 'sparsity'
    # x_axis = 'density'

    # make_time_vs_density_plot(df_time, pathlib.Path(f'./plots/time-vs-{x_axis}.png'), x_axis)
    make_acc_vs_density_plot(df_best, pathlib.Path(f'./plots/accuracy-vs-{x_axis}1.png'), x_axis)



if __name__ == '__main__':
    main()
