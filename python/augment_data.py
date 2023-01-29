#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import argparse
import pathlib
import torch
from testing.datasets import create_cifar10_datasets, create_dataloaders
from testing.numpy_utils import save_eigen_array


def load_models(model: str, datadir: str):
    if model == 'cifar10':
        train_dataset, test_dataset = create_cifar10_datasets(datadir)
        return create_dataloaders(train_dataset, test_dataset, len(train_dataset), len(test_dataset))
    else:
        raise RuntimeError(f'Unknown model {model}')


def main():
    cmdline_parser = argparse.ArgumentParser()
    cmdline_parser.add_argument("--batch-size", help="The batch size", type=int, default=1)
    cmdline_parser.add_argument("--epochs", help="The number of epochs", type=int, default=1)
    cmdline_parser.add_argument('--model', type=str, default='cifar10', help='the data set (default: cifar10)')
    cmdline_parser.add_argument("--seed", help="The initial seed of the random generator", type=int)
    cmdline_parser.add_argument('--datadir', type=str, default='./data', help='the data directory (default: ./data)')
    cmdline_parser.add_argument("--outputdir", type=str, help="the directory where the results are stored")
    args = cmdline_parser.parse_args()

    if args.seed:
        torch.manual_seed(args.seed)

    pathlib.Path(args.outputdir).mkdir(parents=True, exist_ok=True)

    print(f'Generating augmented datasets')
    for epoch in range(args.epochs):
        print(f'epoch {epoch}')
        filename = f'{args.outputdir}/epoch{epoch}.npy'
        train_loader, test_loader = load_models(args.model, args.datadir)
        Xtrain, Ttrain = next(iter(train_loader))
        Xtest, Ttest = next(iter(test_loader))
        print(f'Saving epoch {epoch} data to {filename}')
        with open(filename, "wb") as f:
            save_eigen_array(f, Xtrain.detach().numpy())
            save_eigen_array(f, Ttrain.detach().numpy())
            save_eigen_array(f, Xtest.detach().numpy())
            save_eigen_array(f, Ttest.detach().numpy())
        del Xtrain, Ttrain, Xtest, Ttest


if __name__ == '__main__':
    main()
