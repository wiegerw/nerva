# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import random
from pathlib import Path
from typing import Tuple

import sympy as sp
import torch

from nerva.datasets import load_dict_from_npz


def instantiate_one_hot_rowwise(X: sp.Matrix) -> sp.Matrix:
    m, n = X.shape
    X0 = sp.zeros(m, n)
    for i in range(m):
        j = random.randrange(0, n)
        X0[i, j] = 1

    return X0


class TorchDataLoader(object):
    """
    A data loader with an interface similar to torch.utils.data.DataLoader.
    It produces batches in column layout.
    """

    def __init__(self, Xdata: torch.Tensor, Tdata: torch.IntTensor, batch_size: int):
        """
        :param Xdata: a dataset with row layout
        :param Tdata: an integer vector containing the targets
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


def to_one_hot_torch(x: torch.LongTensor, n_classes: int):
    one_hot = torch.zeros(len(x), n_classes, dtype=torch.float)
    one_hot.scatter_(1, x.unsqueeze(1), 1)
    return one_hot
