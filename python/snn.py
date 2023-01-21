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
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import sparselearning
from sparselearning.core import Masking, CosineDecay
from sparselearning.models import MLP_CIFAR10
from sparselearning.utils import get_mnist_dataloaders, get_cifar10_dataloaders, get_cifar100_dataloaders

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

def setup_logger(args):
    global logger
    if logger == None:
        logger = logging.getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)

    args_copy = copy.deepcopy(args)
    # copy to get a clean hash
    # use the same log file hash if iterations or verbose are different
    # these flags do not change the results
    args_copy.iters = 1
    args_copy.verbose = False
    args_copy.log_interval = 1
    args_copy.seed = 0

    log_path = './logs/{0}_{1}_{2}.log'.format(args.model, args.density, hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def print_and_log(msg):
    global logger
    print(msg)
    logger.info(msg)

def train_model(model, mask, loss_fn, train_loader, lr_scheduler, optimizer, device, epochs, batch_size, log_interval, output_folder):
    model.train()  # Set model in training mode

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = 0
        correct = 0
        n = 0

        # global gradient_norm
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)

            loss = loss_fn(output, target)

            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            n += target.shape[0]

            loss.backward()

            if mask is not None:
                mask.step()
            else:
                optimizer.step()

            if batch_idx % log_interval == 0:
                print_and_log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {}/{} ({:.3f}% '.format(
                    epoch, batch_idx * len(data), len(train_loader) * batch_size,
                           100. * batch_idx / len(train_loader), loss.item(), correct, n, 100. * correct / float(n)))

        # training summary
        print_and_log('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
            'Training summary', train_loss / batch_idx, correct, n, 100. * correct / float(n)))
        lr_scheduler.step()
        print_and_log('Current learning rate: {0}. Time taken for epoch: {1:.2f} seconds.\n'.format(optimizer.param_groups[0]['lr'], time.time() - t0))


def evaluate(model, loss_fn, device, test_loader, is_test_set=False):
    model.eval()
    test_loss = 0
    correct = 0
    n = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            model.t = target
            output = model(data)
            test_loss += loss_fn(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            n += target.shape[0]

    test_loss /= float(n)

    print_and_log('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Test evaluation' if is_test_set else 'Evaluation',
        test_loss, correct, n, 100. * correct / float(n)))
    return correct / float(n)


def load_model(name: str, device) -> torch.nn.Module:
    if name == 'mlp_cifar10':
        return MLP_CIFAR10().to(device)
    raise RuntimeError(f'Unknown model {name}')


def make_loaders(args, dataset: str, validation_split, max_threads) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    if dataset == 'mnist':
        return get_mnist_dataloaders(args.batch_size, args.test_batch_size, validation_split=validation_split)
    elif dataset == 'cifar10':
        return get_cifar10_dataloaders(args.batch_size, args.test_batch_size, validation_split=validation_split, max_threads=max_threads)
    elif dataset == 'cifar100':
        return get_cifar100_dataloaders(args.batch_size, args.test_batch_size, validation_split=validation_split, max_threads=max_threads)
    raise RuntimeError(f'Unknown dataset {dataset}')


def make_optimizer(name, model: nn.Module, lr: float, momentum: float, weight_decay: float, nesterov: bool) -> torch.optim.Optimizer:
    if name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    elif name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise RuntimeError(f'Unknown optimizer: {name}')


def make_mask(args,
              model: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              train_loader
             ) -> Optional[Masking]:
    mask = None
    if args.sparse:
        prune_interval = args.update_frequency if not args.fix else 0
        decay = CosineDecay(args.prune_rate, len(train_loader) * (args.epochs * args.multiplier))
        mask = Masking(optimizer, prune_rate=args.prune_rate, prune_mode=args.prune, prune_rate_decay=decay,
                       growth_mode=args.growth,
                       redistribution_mode=args.redistribution, train_loader=train_loader,
                       prune_interval=prune_interval)
        mask.add_module(model, sparse_init=args.sparse_init, density=args.density)
    return mask


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
    sparselearning.core.add_sparse_args(parser)

    args = parser.parse_args()
    setup_logger(args)
    print_and_log(args)

    device = torch.device("cpu")

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
        print_and_log("\nIteration start: {0}/{1}\n".format(i+1, args.iters))

        train_loader, valid_loader, test_loader = make_loaders(args, args.data, args.valid_split, args.max_threads)
        model = load_model(args.model, device)
        optimizer = make_optimizer(args.optimizer, model, args.lr, args.momentum, args.l2, nesterov=True)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs / 2) * args.multiplier, int(args.epochs * 3 / 4) * args.multiplier], last_epoch=-1)
        mask = make_mask(args, model, optimizer, train_loader)

        print_and_log(model)
        print_and_log('=' * 60)
        print_and_log(args.model)
        print_and_log('=' * 60)
        print_and_log('Prune mode: {0}'.format(args.prune))
        print_and_log('Growth mode: {0}'.format(args.growth))
        print_and_log('Redistribution mode: {0}'.format(args.redistribution))
        print_and_log('=' * 60)

        # create output folder
        output_path = './save/' + str(args.model) + '/' + str(args.data) + '/' + str(args.sparse_init) + '/' + str(args.seed)
        output_folder = os.path.join(output_path, f'sparsity{1 - args.density}' if args.sparse else 'dense')
        if not os.path.exists(output_folder): os.makedirs(output_folder)

        epochs = args.epochs * args.multiplier
        loss_fn = nn.NLLLoss()
        train_model(model, mask, loss_fn, train_loader, lr_scheduler, optimizer, device, epochs, args.batch_size, args.log_interval, output_folder)

        print('Testing model')
        model.load_state_dict(torch.load(os.path.join(output_folder, 'model_final.pth'))['state_dict'])
        evaluate(model, loss_fn, device, test_loader, is_test_set=True)
        print_and_log("\nIteration end: {0}/{1}\n".format(i+1, args.iters))


if __name__ == '__main__':
   main()
