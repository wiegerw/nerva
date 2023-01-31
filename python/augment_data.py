#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import argparse
import pathlib
import sys

import numpy as np
import torch
from testing.datasets import create_cifar10_datasets, create_dataloaders, load_cifar10_data
from testing.numpy_utils import pp


def load_models(model: str, datadir: str):
    if model == 'cifar10':
        train_dataset, test_dataset = create_cifar10_datasets(datadir)
        return create_dataloaders(train_dataset, test_dataset, len(train_dataset), len(test_dataset))
    else:
        raise RuntimeError(f'Unknown model {model}')


def to_eigen(x: np.ndarray):
    if len(x.shape) == 2:
        return x.reshape(x.shape[1], x.shape[0], order='F').T
    return x


def from_eigen(x: np.ndarray):
    if len(x.shape) == 2:
        return x.reshape(x.shape[1], x.shape[0], order='C').T
    return x


def inspect_data(outputdir, epochs):
    for epoch in range(epochs):
        print(f'--- epoch {epoch} ---')
        path = pathlib.Path(outputdir) / f'epoch{epoch}.npz'
        d = np.load(path)
        Xtrain = from_eigen(d['Xtrain'])
        pp(f'Xtrain', Xtrain)


# check if the data is stored correctly
def check(datadir):
    import tempfile
    filename = tempfile.NamedTemporaryFile().name + '_cifar.npz'

    Xtrain, Ttrain, Xtest, Ttest = load_cifar10_data(datadir)
    pp('Xtrain', Xtrain)

    # save the data to .npz
    print(f'Saving data to file {filename}')
    with open(filename, "wb") as f:
        np.savez_compressed(f,
                            Xtrain=to_eigen(Xtrain.detach().numpy()),
                            Ttrain=to_eigen(Ttrain.detach().numpy()),
                            Xtest=to_eigen(Xtest.detach().numpy()),
                            Ttest=to_eigen(Ttest.detach().numpy())
                            )

    # load the .npz data
    print(f'Loading data from file {filename}')
    d = np.load(filename)
    Xtrain_new = from_eigen(d['Xtrain'])
    pp(f'Xtrain_new', Xtrain_new)

    pathlib.Path(filename).unlink()


def main():
    cmdline_parser = argparse.ArgumentParser()
    cmdline_parser.add_argument("--batch-size", help="The batch size", type=int, default=1)
    cmdline_parser.add_argument("--epochs", help="The number of epochs", type=int, default=1)
    cmdline_parser.add_argument('--model', type=str, default='cifar10', help='the data set (default: cifar10)')
    cmdline_parser.add_argument("--seed", help="The initial seed of the random generator", type=int)
    cmdline_parser.add_argument('--datadir', type=str, default='./data', help='the data directory (default: ./data)')
    cmdline_parser.add_argument("--outputdir", type=str, help="the output directory where the results are stored")
    cmdline_parser.add_argument("--inspect", help="if this flag is set, the output directory will be inspected", action="store_true")
    cmdline_parser.add_argument("--check", help="check if the data is the same after saving and loading", action="store_true")
    args = cmdline_parser.parse_args()

    if args.seed:
        torch.manual_seed(args.seed)

    if args.check:
        check(args.datadir)
        sys.exit(0)

    if args.inspect:
        inspect_data(args.outputdir, args.epochs)
        sys.exit(0)

    pathlib.Path(args.outputdir).mkdir(parents=True, exist_ok=True)

    print(f'Generating augmented datasets')
    for epoch in range(args.epochs):
        print(f'epoch {epoch}')
        filename = f'{args.outputdir}/epoch{epoch}.npz'
        train_loader, test_loader = load_models(args.model, args.datadir)
        Xtrain, Ttrain = next(iter(train_loader))
        Xtest, Ttest = next(iter(test_loader))
        print(f'Saving epoch {epoch} data to {filename}')
        with open(filename, "wb") as f:
            np.savez_compressed(f,
                                Xtrain=to_eigen(Xtrain.detach().numpy()),
                                Ttrain=to_eigen(Ttrain.detach().numpy()),
                                Xtest=to_eigen(Xtest.detach().numpy()),
                                Ttest=to_eigen(Ttest.detach().numpy())
                                )
        del Xtrain, Ttrain, Xtest, Ttest


if __name__ == '__main__':
    main()
