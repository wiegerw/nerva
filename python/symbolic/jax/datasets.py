# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from pathlib import Path
from typing import Tuple, Union

import jax
import jax.numpy as jnp


class MemoryDataLoader(object):
    """
    A data loader with an interface similar to torch.utils.data.DataLoader.
    """

    def __init__(self, Xdata: jnp.ndarray, Tdata: jnp.ndarray, batch_size: int, rowwise=True):
        """
        :param Xdata: a dataset with row layout
        :param Tdata: an integer vector containing the targets
        :param batch_size: the batch size
        :param rowwise: determines the layout of the batches (column or row layout)
        """
        self.Xdata = Xdata
        self.Tdata = Tdata
        self.batch_size = batch_size
        self.dataset = Xdata
        self.rowwise = rowwise

    def __iter__(self):
        N = self.Xdata.shape[0]  # N is the number of examples
        K = N // self.batch_size  # K is the number of batches
        for k in range(K):
            batch = range(k * self.batch_size, (k + 1) * self.batch_size)
            if self.rowwise:
                yield self.Xdata[batch], self.Tdata[batch]
            else:
                yield self.Xdata[batch].T, self.Tdata[batch]

    def __len__(self):
        """
        Returns the number of batches
        """
        return self.Xdata.shape[0] // self.batch_size


DataLoader = MemoryDataLoader


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

    data = dict(jnp.load(filename, allow_pickle=True))
    Xtrain, Ttrain, Xtest, Ttest = data['Xtrain'], data['Ttrain'], data['Xtest'], data['Ttest']
    train_loader = MemoryDataLoader(Xtrain, Ttrain, batch_size, rowwise)
    test_loader = MemoryDataLoader(Xtest, Ttest, batch_size, rowwise)
    return train_loader, test_loader
