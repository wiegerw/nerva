#!/usr/bin/env python3

import pathlib
import re
import pandas as pd
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


def make_plot(df: pd.DataFrame, path: pathlib.Path):
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


def main():
    for path in pathlib.Path('.').glob('*.log'):
        df = parse_logfile(path)
        make_plot(df, path)


if __name__ == '__main__':
    main()
