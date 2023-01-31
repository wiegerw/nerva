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



def plot_one_log(df: pd.DataFrame, path: pathlib.Path):
    # Plot 'epoch' vs 'train accuracy'
    plt.plot(df['epoch'], df['train accuracy'], label='train accuracy')

    # Plot 'epoch' vs 'test accuracy'
    plt.plot(df['epoch'], df['test accuracy'], label='test accuracy', color='red')

    # Add labels and title to the plot
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs Epoch ({path})')

    # Add a legend to the plot
    plt.legend()

    # Show the plot
    # plt.show()

    # Save the plot
    plt.savefig(path.with_suffix('.png'))
    plt.close()


def make_acc_vs_density_plot(df: pd.DataFrame, path: pathlib.Path):
    # plot 'accuracy' vs 'density' for each framework using seaborn
    sns.lineplot(x='density', y='test accuracy', hue='framework', data=df)

    # Add labels and title to the plot
    plt.xlabel('Density')
    plt.ylabel('Best test accuracy')
    plt.title(f'Accuracy vs Density')
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

        # change path to start with ./plots instead of ./logs
        # path = pathlib.Path('./plots') / path.name
        # plot_one_log(data, path)


    # make a new df with the best test accuracy for each framework and density and seed
    df_best = df.drop(columns=['epoch', 'time', 'loss'])
    df_best = df_best.groupby(['framework', 'density', 'seed']).max().reset_index()

    make_acc_vs_density_plot(df_best, pathlib.Path('./plots/accuracy-vs-density.png'))



if __name__ == '__main__':
    main()
