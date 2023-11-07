#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torchvision import transforms, datasets


def create_dataloaders(train_dataset, test_dataset, batch_size, test_batch_size) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size,
        num_workers=8,
        pin_memory=True, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        test_batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True)

    return train_loader, test_loader


def extract_tensors_from_dataloader(dataloader: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the dataset and corresponding targets that are wrapped in a data loader
    :param dataloader: a data loader
    """
    dataset = []
    targets = []

    for data_batch, target_batch in dataloader:
        dataset.append(data_batch)
        targets.append(target_batch)

    dataset = torch.cat(dataset, dim=0)
    targets = torch.cat(targets, dim=0)

    return dataset, targets


def create_dataset(data_dir: str, batch_size: int, dataset: str):
    """
    Creates a dataset in .npz format.
    """

    dataset_functions = {
        'cifar10': datasets.CIFAR10,
        'mnist': datasets.MNIST,
    }

    normalize_functions = {
        'cifar10': transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        'mnist': transforms.Normalize((0.1307,), (0.3081,)),
    }

    dataset_function = dataset_functions[dataset]
    normalize_function = normalize_functions[dataset]

    transform = transforms.Compose(
        [transforms.ToTensor(),
         normalize_function,
         transforms.Lambda(lambda x: torch.flatten(x)),
        ]
    )

    train_dataset = dataset_function(data_dir, True, transform=transform, download=True)
    test_dataset = dataset_function(data_dir, False, transform=transform, download=False)
    train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, batch_size, batch_size)
    Xtrain, Ttrain = extract_tensors_from_dataloader(train_loader)
    Xtest, Ttest = extract_tensors_from_dataloader(test_loader)

    filename = Path(data_dir) / f'{dataset}.npz'
    print(f'Saving data to file {filename}')
    with open(filename, "wb") as f:
        np.savez(f,
                 Xtrain=Xtrain.detach().numpy(),
                 Ttrain=Ttrain.detach().numpy(),
                 Xtest=Xtest.detach().numpy(),
                 Ttest=Ttest.detach().numpy()
                )


def create_mnist_dataset(data_dir: str, batch_size: int):
    """
    Creates MNIST train and test datasets without augmentation.
    """

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
        transforms.Lambda(lambda x: torch.flatten(x)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
        transforms.Lambda(lambda x: torch.flatten(x)),
    ])

    train_dataset = datasets.MNIST(data_dir, True, train_transform, download=True)
    test_dataset = datasets.MNIST(data_dir, False, test_transform, download=False)
    train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, batch_size, batch_size)
    Xtrain, Ttrain = extract_tensors_from_dataloader(train_loader)
    Xtest, Ttest = extract_tensors_from_dataloader(test_loader)

    filename = Path(data_dir) / 'mnist.npz'
    print(f'Saving data to file {filename}')
    with open(filename, "wb") as f:
        np.savez_compressed(f,
                            Xtrain=Xtrain.detach().numpy(),
                            Ttrain=Ttrain.detach().numpy(),
                            Xtest=Xtest.detach().numpy(),
                            Ttest=Ttest.detach().numpy()
                            )


def main():
    cmdline_parser = argparse.ArgumentParser()
    cmdline_parser.add_argument("--batch-size", help="The batch size (default: 100)", type=int, default=100)
    cmdline_parser.add_argument('--datadir', type=str, default='./data', help='The directory where data sets are stored (default: ./data)')
    cmdline_parser.add_argument('--dataset', type=str, help='The dataset (cifar10, mnist or all)')
    args = cmdline_parser.parse_args()

    all_datasets = ['cifar10', 'mnist']
    if args.dataset in all_datasets:
        datasets = [args.dataset]
    elif args.dataset == 'all':
        datasets = all_datasets
    else:
        raise RuntimeError(f"Error: dataset '{args.dataset}' is unsupported")

    torch.multiprocessing.set_sharing_strategy('file_system')
    for dataset in datasets:
        create_dataset(args.datadir, args.batch_size, dataset)


if __name__ == '__main__':
    main()
