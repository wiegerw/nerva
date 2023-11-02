# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from pathlib import Path
from typing import Tuple, Union

import torch

from utilities import load_dict_from_npz


def to_one_hot_rowwise(x: torch.LongTensor, num_classes: int):
    one_hot = torch.zeros(len(x), num_classes, dtype=torch.float)
    one_hot.scatter_(1, x.unsqueeze(1), 1)
    return one_hot


def to_one_hot_colwise(x: torch.LongTensor, num_classes: int):
    one_hot = torch.zeros(num_classes, len(x), dtype=torch.float)
    one_hot.scatter_(0, x.unsqueeze(0), 1)
    return one_hot


class MemoryDataLoader(object):
    """
    A data loader with an interface similar to torch.utils.data.DataLoader.
    """

    def __init__(self, Xdata: torch.Tensor, Tdata: torch.IntTensor, batch_size: int, rowwise=True, num_classes=0):
        """
        :param Xdata: a dataset with row layout
        :param Tdata: the expected targets. In case of a classification task the targets may be specified as a vector
                      of integers that denote the expected classes. In such a case the targets will be expanded on the
                      fly using one hot encoding.
        :param batch_size: the batch size
        :param rowwise: determines the layout of the batches (column or row layout)
        :param num_classes: the number of classes in case of a classification problem, 0 otherwise
        """
        self.Xdata = Xdata
        self.Tdata = Tdata
        self.batch_size = batch_size
        self.dataset = Xdata
        self.rowwise = rowwise
        self.num_classes = int(Tdata.max() + 1) if num_classes == 0 and len(Tdata.shape) == 1 else num_classes

    def __iter__(self):
        N = self.Xdata.shape[0]  # N is the number of examples
        K = N // self.batch_size  # K is the number of batches
        for k in range(K):
            batch = range(k * self.batch_size, (k + 1) * self.batch_size)
            if self.rowwise:
                yield self.Xdata[batch], to_one_hot_rowwise(self.Tdata[batch], self.num_classes) if self.num_classes else self.Tdata[batch]
            else:
                yield self.Xdata[batch].T, to_one_hot_colwise(self.Tdata[batch], self.num_classes) if self.num_classes else self.Tdata[batch]

    def __len__(self):
        """
        Returns the number of batches
        """
        return self.Xdata.shape[0] // self.batch_size


DataLoader = Union[MemoryDataLoader, torch.utils.data.DataLoader]


def create_npz_dataloaders(filename: str, batch_size: int, rowwise=True) -> Tuple[MemoryDataLoader, MemoryDataLoader]:
    """
    Creates a data loader from a file containing a dictionary with Xtrain, Ttrain, Xtest and Ttest tensors
    :param filename: a file in NumPy .npz format
    :param batch_size: the batch size of the data loader
    :param rowwise: determines the layout of the batches (column or row layout)
    :return: a tuple of data loaders
    """
    path = Path(filename)
    print(f'Loading dataset from file {path}')
    if not path.exists():
        raise RuntimeError(f"Could not load file '{path}'")

    data = load_dict_from_npz(filename)
    Xtrain, Ttrain, Xtest, Ttest = data['Xtrain'], data['Ttrain'], data['Xtest'], data['Ttest']
    train_loader = MemoryDataLoader(Xtrain, Ttrain, batch_size, rowwise)
    test_loader = MemoryDataLoader(Xtest, Ttest, batch_size, rowwise)
    return train_loader, test_loader
