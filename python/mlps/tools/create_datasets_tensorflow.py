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
        np.savez_compressed(f,
                            Xtrain=X_train.astype(np.float32),
                            Ttrain=T_train.astype(np.int64),
                            Xtest=X_test.astype(np.float32),
                            Ttest=T_test.astype(np.int64))


def normalize_and_flatten(data):
    data = data / 255.0
    data = data.reshape(data.shape[0], -1)
    return data


def load_cifar10(root: str):
    (X_train, T_train), (X_test, T_test) = cifar10.load_data()
    X_train = normalize_and_flatten(X_train)
    T_train = T_train.ravel()
    X_test = normalize_and_flatten(X_test)
    T_test = T_test.ravel()
    save_dataset(Path(root) / 'cifar10.npz', X_train, T_train, X_test, T_test)


def load_mnist(root: str):
    (X_train, T_train), (X_test, T_test) = mnist.load_data()
    X_train = normalize_and_flatten(X_train)
    T_train = T_train.ravel()
    X_test = normalize_and_flatten(X_test)
    T_test = T_test.ravel()
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
