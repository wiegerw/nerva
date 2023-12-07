#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST


def save_dataset(path: Path, X_train, T_train, X_test, T_test):
    """Saves the dataset in a dictionary in .npz format."""
    print(f'Saving data to file {path}')
    with open(path, "wb") as f:
        np.savez_compressed(f, Xtrain=X_train, Ttrain=T_train, Xtest=X_test, Ttest=T_test)


def load_cifar10(root: str, batch_size=64, num_workers=1):
    """ Downloads the CIFAR-10 dataset, normalizes and flattens it, and saves it into .npz format."""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = CIFAR10(root=root, train=True, transform=transform, download=True)
    test_dataset = CIFAR10(root=root, train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    flatten = lambda x: x.view(x.size(0), -1)

    X_train, T_train = zip(*[(flatten(x), t) for x, t in train_loader])
    X_train = torch.cat(X_train)
    T_train = torch.cat(T_train)

    X_test, T_test = zip(*[(flatten(x), t) for x, t in test_loader])
    X_test = torch.cat(X_test)
    T_test = torch.cat(T_test)

    save_dataset(Path(root) / 'cifar10.npz', X_train, T_train, X_test, T_test)


def load_mnist(root: str, batch_size=64, num_workers=1):
    """ Downloads the MNIST dataset, normalizes and flattens it, and saves it into .npz format."""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5,))])

    train_dataset = MNIST(root=root, train=True, transform=transform, download=True)
    test_dataset = MNIST(root=root, train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    X_train, T_train = zip(*[(x.view(x.size(0), -1), t) for x, t in train_loader])
    X_train = torch.cat(X_train)
    T_train = torch.cat(T_train)

    X_test, T_test = zip(*[(x.view(x.size(0), -1), t) for x, t in test_loader])
    X_test = torch.cat(X_test)
    T_test = torch.cat(T_test)

    save_dataset(Path(root) / 'mnist.npz', X_train, T_train, X_test, T_test)


def main():
    cmdline_parser = argparse.ArgumentParser()
    cmdline_parser.add_argument("--batch-size", help="The batch size (default: 100)", type=int, default=100)
    cmdline_parser.add_argument("--workers", help="The number of workers used for loading (default: 1)", type=int, default=1)
    cmdline_parser.add_argument('--root', type=str, default='./data', help='The directory where data sets are stored (default: ./data)')
    cmdline_parser.add_argument('dataset', type=str, help='The dataset (cifar10 or mnist)')
    args = cmdline_parser.parse_args()

    torch.multiprocessing.set_sharing_strategy('file_system')
    if args.dataset == 'cifar10':
        load_cifar10(args.root, args.batch_size, args.workers)
    elif args.dataset == 'mnist':
        load_mnist(args.root, args.batch_size, args.workers)
    else:
        raise RuntimeError(f"Unknown dataset '{args.dataset}'")


if __name__ == '__main__':
    main()
