#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import argparse
import pickle
import struct
from pathlib import Path

import numpy as np


def save_dataset(path: Path, X_train, T_train, X_test, T_test):
    """Saves the dataset in a dictionary in .npz format."""
    print(f'Saving data to file {path}')
    with open(path, "wb") as f:
        np.savez_compressed(f, Xtrain=X_train, Ttrain=T_train, Xtest=X_test, Ttest=T_test)


def normalize_and_flatten(data):
    data = data / 255.0
    data = data.reshape(data.shape[0], -1)
    return data


def create_cifar10(root: str, cifar10_folder: str):
    def unpickle(path: Path):
        with open(path, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        return data

    train_data = []
    train_labels = []

    for i in range(1, 6):
        batch_data = unpickle(Path(cifar10_folder) / f'data_batch_{i}')
        train_data.append(batch_data[b'data'])
        train_labels.extend(batch_data[b'labels'])

    X_train = np.vstack(train_data)
    T_train = np.array(train_labels)

    test_data = unpickle(Path(cifar10_folder) / 'test_batch')
    X_test = test_data[b'data']
    T_test = np.array(test_data[b'labels'])

    X_train = normalize_and_flatten(X_train)
    X_test = normalize_and_flatten(X_test)

    save_dataset(Path(root) / 'cifar10.npz', X_train, T_train, X_test, T_test)


def create_mnist(root: str, mnist_folder: str):
    def read_idx(path):
        with open(path, 'rb') as f:
            magic, num_items = struct.unpack(">II", f.read(8))
            if magic == 2051:
                rows, cols = struct.unpack(">II", f.read(8))
                data = np.fromfile(f, dtype=np.uint8)
                data = data.reshape(num_items, rows, cols).astype(np.float32)
            elif magic == 2049:
                data = np.fromfile(f, dtype=np.uint8)
                data = data.astype(np.int64)
            return data

    X_train = read_idx(Path(mnist_folder) / "train-images-idx3-ubyte")
    T_train = read_idx(Path(mnist_folder) / "train-labels-idx1-ubyte")
    X_test = read_idx(Path(mnist_folder) / "t10k-images-idx3-ubyte")
    T_test = read_idx(Path(mnist_folder) / "t10k-labels-idx1-ubyte")

    X_train = normalize_and_flatten(X_train)
    X_test = normalize_and_flatten(X_test)

    save_dataset(Path(root) / 'mnist.npz', X_train, T_train, X_test, T_test)


def main():
    cmdline_parser = argparse.ArgumentParser()
    cmdline_parser.add_argument('--root', type=str, default='./data', help='The directory where data sets are stored (default: ./data)')
    cmdline_parser.add_argument('--mnist-folder', type=str, default='./data/MNIST/raw', help='The directory where MNIST data is located (default: ./data/MNIST/raw)')
    cmdline_parser.add_argument('--cifar10-folder', type=str, default='./data/cifar-10-batches-py', help='The directory where MNIST data is located (default: ./data/cifar-10-batches-py)')
    cmdline_parser.add_argument('dataset', type=str, help='The dataset (cifar10 or mnist)')
    args = cmdline_parser.parse_args()

    if args.dataset == 'cifar10':
        create_cifar10(args.root, args.cifar10_folder)
    elif args.dataset == 'mnist':
        create_mnist(args.root, args.mnist_folder)
    else:
        raise RuntimeError(f"Unknown dataset '{args.dataset}'")


if __name__ == '__main__':
    main()
