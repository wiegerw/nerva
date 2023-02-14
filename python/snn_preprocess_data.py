#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import argparse
import pathlib
import sys

import numpy as np
import torch
from testing.datasets import create_cifar10_augmented_datasets, create_dataloaders, custom_load_cifar10_data, \
    create_cifar10_augmented_dataloaders, extract_tensors_from_dataloader, save_train_test_data_to_npz
from testing.numpy_utils import pp


def load_models(model: str, datadir: str):
    if model == 'cifar10':
        train_dataset, test_dataset = create_cifar10_augmented_datasets(datadir)
        return create_dataloaders(train_dataset, test_dataset, len(train_dataset), len(test_dataset))
    else:
        raise RuntimeError(f'Unknown model {model}')


def inspect_data(outputdir, epochs):
    for epoch in range(epochs):
        print(f'--- epoch {epoch} ---')
        path = pathlib.Path(outputdir) / f'epoch{epoch}.npz'
        d = np.load(path)
        Xtrain = d['Xtrain']
        pp(f'Xtrain', Xtrain)



# check if the data is stored correctly
def check(datadir):
    import tempfile
    from nervalib import data_set

    Xtrain, Ttrain, Xtest, Ttest = custom_load_cifar10_data(datadir)
    pp('Xtrain', Xtrain)
    pp('Ttrain', Ttrain)

    filename = tempfile.NamedTemporaryFile().name + '_cifar.npz'

    # save the data to .npz
    print(f'Saving data to file {filename}')
    with open(filename, "wb") as f:
        np.savez_compressed(f,
                            Xtrain=Xtrain.detach().numpy(),
                            Ttrain=Ttrain.detach().numpy(),
                            Xtest=Xtest.detach().numpy(),
                            Ttest=Ttest.detach().numpy()
                            )

    # load the .npz data
    print(f'Loading data from file {filename}')
    d = np.load(filename)
    Xtrain_new = d['Xtrain']
    pp(f'Xtrain_new', Xtrain_new)

    print(f'Loading data to c++ data_set {filename}')
    data2 = data_set()
    data2.import_cifar10_from_npz(filename)
    data2.info()

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
        train_loader, test_loader = create_cifar10_augmented_dataloaders(args.batch_size, args.batch_size, args.datadir)
        Xtrain, Ttrain = extract_tensors_from_dataloader(train_loader)
        Xtest, Ttest = extract_tensors_from_dataloader(test_loader)
        print(f'Saving epoch {epoch} data to {filename}')
        save_train_test_data_to_npz(filename, Xtrain, Ttrain, Xtest, Ttest)
        del Xtrain, Ttrain, Xtest, Ttest


if __name__ == '__main__':
    main()
