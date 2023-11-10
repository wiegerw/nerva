#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import argparse
from pathlib import Path

import numpy as np
from keras.datasets import cifar10, mnist


def save_dataset(path: Path, X_train, T_train, X_test, T_test):
    """Saves the dataset in a dictionary in .npz format."""
    print(f'Saving data to file {path}')
    with open(path, "wb") as f:
        np.savez_compressed(f, Xtrain=X_train, Ttrain=T_train, Xtest=X_test, Ttest=T_test)


def load_cifar10(root: str):
    (X_train, T_train), (X_test, T_test) = cifar10.load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    save_dataset(Path(root) / 'cifar10.npz', X_train, T_train, X_test, T_test)


def load_mnist(root: str):
    (X_train, T_train), (X_test, T_test) = mnist.load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    save_dataset(Path(root) / 'mnist.npz', X_train, T_train, X_test, T_test)


def main():
    cmdline_parser = argparse.ArgumentParser()
    cmdline_parser.add_argument('--root', type=str, default='./data', help='The directory where data sets are stored (default: ./data)')
    cmdline_parser.add_argument('dataset', type=str, help='The dataset (cifar10 or mnist)')
    args = cmdline_parser.parse_args()

    if args.dataset == 'cifar10':
        load_cifar10(args.root)
    elif args.dataset == 'mnist':
        load_mnist(args.root)
    else:
        raise RuntimeError(f"Unknown dataset '{args.dataset}'")


if __name__ == '__main__':
    main()
