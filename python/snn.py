# This code is originally from https://github.com/VITA-Group/Random_Pruning
# There is no license information available

from __future__ import print_function

import os
import time
import argparse
import logging
import hashlib
import copy
import random
from typing import Tuple

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import sparselearning
from sparselearning.core import Masking
from sparselearning.utils import get_mnist_dataloaders, get_cifar10_dataloaders, get_cifar100_dataloaders
from sparselearning import train_nerva, train_pytorch
from sparselearning.logger import DefaultLogger

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
cudnn.benchmark = True
cudnn.deterministic = True

if not os.path.exists('./models'): os.mkdir('./models')
if not os.path.exists('./logs'): os.mkdir('./logs')
logger = None

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    print("SAVING")
    torch.save(state, filename)


def make_loaders(args, dataset: str, validation_split, max_threads) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    if dataset == 'mnist':
        return get_mnist_dataloaders(args.batch_size, args.test_batch_size, validation_split=validation_split)
    elif dataset == 'cifar10':
        return get_cifar10_dataloaders(args.batch_size, args.test_batch_size, validation_split=validation_split, max_threads=max_threads)
    elif dataset == 'cifar100':
        return get_cifar100_dataloaders(args.batch_size, args.test_batch_size, validation_split=validation_split, max_threads=max_threads)
    raise RuntimeError(f'Unknown dataset {dataset}')


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N', help='input batch size for testing (default: 100)')
    parser.add_argument('--multiplier', type=int, default=1, metavar='N', help='extend training time by multiplier times')
    parser.add_argument('--epochs', type=int, default=160, metavar='N', help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=17, metavar='S', help='random seed (default: 17)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--optimizer', type=str, default='sgd', help='The optimizer to use. Default: sgd. Options: sgd, adam.')
    randomhash = ''.join(str(time.time()).split('.'))
    parser.add_argument('--save', type=str, default=randomhash + '.pt', help='path to save the final model')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--decay_frequency', type=int, default=25000)
    parser.add_argument('--l1', type=float, default=0.0)
    parser.add_argument('--valid_split', type=float, default=0.1)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--l2', type=float, default=5.0e-4)
    parser.add_argument('--iters', type=int, default=1, help='How many times the model should be run after each other. Default=1')
    parser.add_argument('--save-features', action='store_true', help='Resumes a saved model and saves its feature data to disk for plotting.')
    parser.add_argument('--bench', action='store_true', help='Enables the benchmarking of layers and estimates sparse speedups')
    parser.add_argument('--max-threads', type=int, default=10, help='How many threads to use for data loading.')
    parser.add_argument('--scaled', action='store_true', help='scale the initialization by 1/density')
    parser.add_argument('--nerva', action='store_true', help='use the Nerva library')
    sparselearning.core.add_sparse_args(parser)

    args = parser.parse_args()
    print_and_log = DefaultLogger(args)
    print_and_log(str(args))

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print_and_log('\n\n')
    print_and_log('='*80)

    # fix random seed for Reproducibility
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    for i in range(args.iters):
        print_and_log("\nIteration start: {0}/{1}\n".format(i + 1, args.iters))
        train_loader, valid_loader, test_loader = make_loaders(args, args.data, args.valid_split, args.max_threads)
        if args.nerva:
            train_nerva.train_and_test(i, args, device, train_loader, test_loader, print_and_log)
        else:
            train_pytorch.train_and_test(i, args, device, train_loader, test_loader, print_and_log)


if __name__ == '__main__':
   main()
