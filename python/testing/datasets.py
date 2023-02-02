import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision
from torch.nn import functional as F
from torchvision import transforms, datasets

from testing.numpy_utils import flatten_numpy, normalize_image_data, pp


def create_cifar10_augmented_datasets(datadir='./data'):
    """Creates train and test datasets with augmentation."""

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
    """Creates train and test datasets without augmentation."""

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


def create_dataloaders(train_dataset, test_dataset, batch_size, test_batch_size):
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
    def __init__(self, Xdata: torch.Tensor, Tdata: torch.Tensor, batch_size: int):
        self.Xdata = Xdata
        self.Tdata = Tdata
        self.batch_size = batch_size
        self.dataset = Xdata  # for conformance to the DataLoader interface

    def __iter__(self):
        N = self.Xdata.shape[0]  # N is the number of examples
        K = N // self.batch_size  # K is the number of batches
        for k in range(K):
            batch = range(k * self.batch_size, (k + 1) * self.batch_size)
            yield self.Xdata[batch], self.Tdata[batch]

    # returns the number of batches
    def __len__(self):
        return self.Xdata.shape[0] // self.batch_size


def extract_tensors_from_dataloader(dataloader) -> Tuple[torch.Tensor, torch.Tensor]:
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

    Xtrain, Ttrain, Xtest, Ttest = load_data_from_npz(filename)
    train_loader = TorchDataLoader(Xtrain, Ttrain, batch_size)
    test_loader = TorchDataLoader(Xtest, Ttest, batch_size)
    return train_loader, test_loader


def to_eigen(x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 2:
        return x.reshape(x.shape[1], x.shape[0], order='F').T
    return x


def from_eigen(x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 2:
        return x.reshape(x.shape[1], x.shape[0], order='C').T
    return x


# save data to file in a format readable in C++
def save_data_to_npz(filename, Xtrain: torch.Tensor, Ttrain: torch.Tensor, Xtest: torch.Tensor, Ttest: torch.Tensor):
    print(f'Saving data to {filename}')
    with open(filename, "wb") as f:
        np.savez_compressed(f,
                            Xtrain=to_eigen(Xtrain.detach().numpy()),
                            Ttrain=to_eigen(Ttrain.detach().numpy()),
                            Xtest=to_eigen(Xtest.detach().numpy()),
                            Ttest=to_eigen(Ttest.detach().numpy())
                            )


# load data from file in a format readable in C++
def load_data_from_npz(filename):
    print(f'Loading data from {filename}')
    d = np.load(filename, allow_pickle=True)
    Xtrain = torch.Tensor(from_eigen(d['Xtrain']))
    Ttrain = torch.LongTensor(from_eigen(d['Ttrain']))
    Xtest = torch.Tensor(from_eigen(d['Xtest']))
    Ttest = torch.LongTensor(from_eigen(d['Ttest']))
    return Xtrain, Ttrain, Xtest, Ttest
