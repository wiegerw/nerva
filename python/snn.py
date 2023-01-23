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

<<<<<<< HEAD
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

=======
>>>>>>> 5418887 (Made some preparations for training using Nerva in SNN experiment)

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
