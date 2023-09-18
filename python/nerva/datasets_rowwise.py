# Copyright 2022 - 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import os
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torchvision
from torch.nn import functional as F
from torchvision import transforms, datasets

import nervalibrowwise
from nerva.utilities import flatten_numpy


class DataSet(nervalibrowwise.DataSetView):
    def __init__(self, Xtrain, Ttrain, Xtest, Ttest):
        super().__init__(Xtrain.T, Ttrain.T, Xtest.T, Ttest.T)
        # store references to the original data to make sure it is not destroyed
        self.keep_alive = [Xtrain, Ttrain, Xtest, Ttest]


def create_cifar10_augmented_datasets(datadir='./data'):
    """
    Creates train and test datasets with augmentation.
    """

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(x.unsqueeze(0), (4, 4, 4, 4), mode='reflect').squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        transforms.Lambda(lambda x: torch.flatten(x)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
        transforms.Lambda(lambda x: torch.flatten(x)),
    ])

    train_dataset = datasets.CIFAR10(datadir, True, train_transform, download=True)
    test_dataset = datasets.CIFAR10(datadir, False, test_transform, download=False)

    return train_dataset, test_dataset


def create_cifar10_datasets(datadir='./data'):
    """
    Creates train and test datasets without augmentation.
    """

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))

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

    train_dataset = datasets.CIFAR10(datadir, True, train_transform, download=True)
    test_dataset = datasets.CIFAR10(datadir, False, test_transform, download=False)

    return train_dataset, test_dataset


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


def create_cifar10_augmented_dataloaders(batch_size, test_batch_size, datadir):
    train_dataset, test_dataset = create_cifar10_augmented_datasets(datadir=datadir)
    return create_dataloaders(train_dataset, test_dataset, batch_size, test_batch_size)


def create_cifar10_dataloaders(batch_size, test_batch_size, datadir):
    train_dataset, test_dataset = create_cifar10_datasets(datadir=datadir)
    train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, batch_size, test_batch_size)
    return train_loader, test_loader


def normalize_image_data(X: np.array, mean=None, std=None):
    if not mean:
        mean = X.mean(axis=(0, 1, 2))
    if not std:
        std = X.std(axis=(0, 1, 2))
    return (X  - mean) / std


def custom_load_cifar10_data(datadir):
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    trainset = torchvision.datasets.CIFAR10(root=datadir, train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root=datadir, train=False, download=True)

    Xtrain = normalize_image_data(trainset.data / 255, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    Xtrain = flatten_numpy(Xtrain)
    Xtrain = torch.Tensor(Xtrain)
    Ttrain = torch.LongTensor(trainset.targets)

    Xtest = normalize_image_data(testset.data / 255)
    Xtest = flatten_numpy(Xtest)
    Xtest = torch.Tensor(Xtest)
    Ttest = torch.LongTensor(testset.targets)

    return Xtrain, Ttrain, Xtest, Ttest


class TorchDataLoader(object):
    """
    A data loader with an interface similar to torch.utils.data.DataLoader.
    """

    def __init__(self, Xdata: torch.Tensor, Tdata: torch.LongTensor, batch_size: int):
        """
        :param Xdata: a dataset
        :param Tdata: the targets for the dataset
        :param batch_size: the batch size
        """
        self.Xdata = Xdata
        self.Tdata = Tdata
        self.batch_size = batch_size
        self.dataset = Xdata

    def __iter__(self):
        N = self.Xdata.shape[0]  # N is the number of examples
        K = N // self.batch_size  # K is the number of batches
        for k in range(K):
            batch = range(k * self.batch_size, (k + 1) * self.batch_size)
            yield self.Xdata[batch], self.Tdata[batch]

    def __len__(self):
        """
        Returns the number of batches
        """
        return self.Xdata.shape[0] // self.batch_size


DataLoader = Union[TorchDataLoader, torch.utils.data.DataLoader]


def extract_tensors_from_dataloader(dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the dataset and corresponding targets that are wrapped in a data loader
    :param dataloader: a data loader
    """
    dataset = dataloader.dataset
    batch_size = len(dataset) // len(dataloader)

    Xdata = torch.empty((len(dataset),) + dataset[0][0].shape)
    Tdata = torch.empty(len(dataset), dtype=torch.long)

    for i, (X, T) in enumerate(dataloader):
        Xdata[i * batch_size: (i + 1) * batch_size] = X
        Tdata[i * batch_size: (i + 1) * batch_size] = T

    return Xdata, Tdata


def create_npz_dataloaders(filename: str, batch_size: int) -> Tuple[TorchDataLoader, TorchDataLoader]:
    """
    Creates a data loader from a file containing a dictionary with Xtrain, Ttrain, Xtest and Ttest tensors
    :param filename: a file in NumPy .npz format
    :param batch_size: the batch size of the data loader
    :return: a tuple of data loaders
    """
    path = Path(filename)
    print(f'Loading dataset from file {path}')
    if not path.exists():
        raise RuntimeError(f"Could not load file '{path}'")

    data = load_dict_from_npz(filename)
    Xtrain, Ttrain, Xtest, Ttest = data['Xtrain'], data['Ttrain'], data['Xtest'], data['Ttest']
    train_loader = TorchDataLoader(Xtrain, Ttrain, batch_size)
    test_loader = TorchDataLoader(Xtest, Ttest, batch_size)
    return train_loader, test_loader


def save_dict_to_npz(filename: str, data: dict[str, torch.Tensor]) -> None:
    """
    Saves a dictionary to file in .npz format
    :param filename: a file name
    :param data: a dictionary
    """
    data = {key: value.detach().numpy() for key, value in data.items()}
    with open(filename, "wb") as f:
        np.savez_compressed(f, **data)


def load_dict_from_npz(filename: str) -> Dict[str, Union[torch.Tensor, torch.LongTensor]]:
    """
    Loads a dictionary from a file in .npz format
    :param filename: a file name
    :return: a dictionary
    """
    def make_tensor(x: np.ndarray) -> Union[torch.Tensor, torch.LongTensor]:
        if np.issubdtype(x.dtype, np.integer):
            return torch.LongTensor(x)
        return torch.Tensor(x)

    data = dict(np.load(filename, allow_pickle=True))
    data = {key: make_tensor(value) for key, value in data.items()}
    return data
