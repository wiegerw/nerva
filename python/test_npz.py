#!/usr/bin/env python3

from unittest import TestCase

import torch

from testing.datasets import create_cifar10_dataloaders, extract_tensors_from_dataloader, save_train_test_data_to_npz, \
    load_train_test_data_from_npz
from testing.numpy_utils import pp


class TestNPZ(TestCase):
    def test_load_save(self):
        batch_size = 100
        datadir = './data'
        filename = 'test_npz.npz'
        train_loader, test_loader = create_cifar10_dataloaders(batch_size, batch_size, datadir)
        Xtrain, Ttrain = extract_tensors_from_dataloader(train_loader)
        Xtest, Ttest = extract_tensors_from_dataloader(test_loader)

        pp('Xtrain', Xtrain)
        pp('Ttrain', Ttrain)
        pp('Xtest', Xtest)
        pp('Ttest', Ttest)

        save_train_test_data_to_npz(filename, Xtrain, Ttrain, Xtest, Ttest)
        Xtrain1, Ttrain1, Xtest1, Ttest1 = load_train_test_data_from_npz(filename)

        pp('Xtrain1', Xtrain1)
        pp('Ttrain1', Ttrain1)
        pp('Xtest1', Xtest1)
        pp('Ttest1', Ttest1)

        self.assertTrue(torch.equal(Xtrain1, Xtrain))
        self.assertTrue(torch.equal(Ttrain1, Ttrain))
        self.assertTrue(torch.equal(Xtest1, Xtest))
        self.assertTrue(torch.equal(Ttest1, Ttest))


if __name__ == '__main__':
    import unittest
    unittest.main()
