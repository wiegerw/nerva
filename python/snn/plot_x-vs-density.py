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


def get_full_dataframe(folder: pathlib.Path = pathlib.Path('.')) -> pd.DataFrame:
    df = pd.DataFrame()
    for path in folder.glob('*.log'):
        data = parse_logfile(path)

        # example path.name = 'nerva-dense-augmented-seed1.log'
        # example path.name = 'nerva-sparse-0.1-augmented-seed1.log'
        if 'dense' in path.name:
            density = 1.0
        else:
            density = float(path.name.split('-')[2])
        framework = path.name.split('-')[0]
        seed = int(path.name.split('-')[-1].split('.')[0].replace('seed', ''))

        data['framework'] = framework
        data['density'] = density
        data['seed'] = seed

        df = pd.concat([df, data])
    return df


def make_df_time(df: pd.DataFrame):
    # make a new df with the total time summed for each framework and density and seed
    df_time = df.drop(columns=['epoch', 'lr', 'loss', 'train accuracy', 'test accuracy'])
    df_time = df_time.groupby(['framework', 'density', 'seed']).sum().reset_index()
    df_time['time'] = df_time['time'] / 60  # convert time to minutes
    df_time['sparsity'] = 1 - df_time['density']
    return df_time


def make_df_acc(df: pd.DataFrame):
    # make a new df with the best test accuracy for each framework and density and seed
    df_best = df.drop(columns=['epoch', 'time', 'loss'])
    df_best = df_best.groupby(['framework', 'density', 'seed']).max().reset_index()
    df_best['sparsity'] = 1 - df_best['density']
    return df_best


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


def make_acc_vs_density_plot(df: pd.DataFrame, path: pathlib.Path, x_axis: str = 'density', log_scale: bool = False):
    palette = sns.color_palette()
    frameworks = {'nerva': {'label': 'Nerva', 'color': palette[0]},
                  'torch': {'label': 'PyTorch', 'color': palette[1]}}
    for fw in frameworks:
        runs = (df['framework'] == fw) & (df['sparsity'] != 0)  # & (df['sparsity'] != 0.5)
        sns.lineplot(x=x_axis, y='test accuracy', data=df[runs], marker='o',
                     label=frameworks[fw]['label'], color=frameworks[fw]['color'])

        if x_axis == 'sparsity':
            # horizontal lines for the fully dense model
            runs = (df['framework'] == fw) & (df['density'] == 1.0)
            tmp_df = df[runs].copy(deep=True)
            tmp_df = pd.concat([tmp_df, tmp_df], ignore_index=True)
            tmp_df.loc[:int(len(tmp_df)/2)-1, 'sparsity'] = 0.5
            tmp_df.loc[int(len(tmp_df)/2):, 'sparsity'] = 0.999
            sns.lineplot(data=tmp_df, x=x_axis, y='test accuracy', linestyle='--',
                         label=f"{frameworks[fw]['label']} dense", color=frameworks[fw]['color'])

    plt.grid(zorder=0)
    plt.ylim(ymin=0)
    if log_scale:
        set_log_scale(x_axis)
    # Add labels and title to the plot
    xlabel = x_axis[0].upper() + x_axis[1:]  # make first letter x_axis uppercase
    plt.xlabel(xlabel)
    plt.ylabel('Best test accuracy')
    plt.title(f'Accuracy vs {xlabel}')
    # Add a legend to the plot. Make sure nerva is presented as Nerva and torch as PyTorch
    plt.legend()
    # Save the plot
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def make_time_vs_density_plot(df: pd.DataFrame, path: pathlib.Path, x_axis: str = 'density', log_scale: bool = False):
    palette = sns.color_palette()
    frameworks = {'nerva': {'label': 'Nerva', 'color': palette[0]},
                  'torch': {'label': 'PyTorch', 'color': palette[1]}}
    for fw in frameworks:
        # remove sparsity=0 from df
        runs = (df['framework'] == fw) & (df['sparsity'] != 0)  # & (df['sparsity'] != 0.5)
        sns.lineplot(x=x_axis, y='time', data=df[runs], marker='o',
                     label=frameworks[fw]['label'], color=frameworks[fw]['color'])

        # horizontal lines for the fully dense model
        runs = (df['framework'] == fw) & (df['density'] == 1.0)
        tmp_df = df[runs].copy(deep=True)
        tmp_df = pd.concat([tmp_df, tmp_df], ignore_index=True)
        tmp_df.loc[:int(len(tmp_df)/2)-1, 'sparsity'] = 0.5
        tmp_df.loc[int(len(tmp_df)/2):, 'sparsity'] = 0.999
        sns.lineplot(data=tmp_df, x=x_axis, y='time', linestyle='--',
                     label=f"{frameworks[fw]['label']} dense", color=frameworks[fw]['color'])

    plt.grid(zorder=0)
    plt.ylim(ymin=0)
    if log_scale:
        set_log_scale(x_axis)
    # Add labels and title to the plot
    xlabel = x_axis[0].upper() + x_axis[1:]  # make first letter x_axis uppercase
    plt.xlabel(xlabel)
    plt.ylabel('Training time (minutes)')
    plt.title(f'Time vs {xlabel}')
    # Add a legend to the plot
    plt.legend()
    # Save the plot
    plt.savefig(path, bbox_inches='tight')
    plt.close()



def main():
    # folder = pathlib.Path('./logs')
    folder = pathlib.Path('./seed12')
    df = get_full_dataframe(folder)

    df_time = make_df_time(df)
    df_acc = make_df_acc(df)


    x_axis = 'sparsity'
    # x_axis = 'density'

    # make_time_vs_density_plot(df_time, pathlib.Path(f'./seed12_plots/time-vs-{x_axis}.pdf'), x_axis)  # log_scale=True)
    make_acc_vs_density_plot(df_acc, pathlib.Path(f'./seed12_plots/accuracy-vs-{x_axis}.pdf'), x_axis, log_scale=True)



if __name__ == '__main__':
    main()
