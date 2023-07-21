#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import argparse
import random
import re
import shlex
import sys
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Linear

from nerva.datasets import create_npz_dataloaders, create_cifar10_augmented_dataloaders, create_cifar10_dataloaders, \
    save_dict_to_npz, load_dict_from_npz
from nerva.training import compute_sparse_layer_densities, print_epoch
from nerva.utilities import StopWatch, pp, parse_function_call


def reservoir_sample(k: int, n: int) -> List[int]:
    # Initialize the reservoir with the first k elements
    reservoir = [i for i in range(k)]

    # Iterate over the remaining elements
    for i in range(k, n):
        j = random.randint(0, i)
        if j < k:
            reservoir[j] = i

    return reservoir


def create_mask(W: torch.Tensor, non_zero_count: int) -> torch.Tensor:
    """
    Creates a boolean matrix with the same shape as W, and with exactly non_zero_count positions equal to 1
    :param W:
    :param non_zero_count:
    :return:
    """
    mask = torch.zeros_like(W)

    I = reservoir_sample(non_zero_count, W.numel())  # generates non_zero_count random indices in W
    for i in I:
        row = i // W.size(1)
        col = i % W.size(1)
        mask[row, col] = 1

    return mask


class MLPPyTorch(nn.Module):
    """ PyTorch Multilayer perceptron that supports sparse layers using binary masks.
    """
    def __init__(self, layer_sizes, layer_densities):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.optimizer = None
        self.masks = None
        n = len(layer_sizes) - 1  # the number of layers
        self.layers = nn.ModuleList()
        for i in range(n):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        self._set_masks(layer_densities)

    def _set_masks(self, layer_densities):
        self.masks = []
        for layer, density in zip(self.layers, layer_densities):
            if density == 1.0:
                self.masks.append(None)
            else:
                self.masks.append(create_mask(layer.weight, round(density * layer.weight.numel())))

    def apply_masks(self):
        for layer, mask in zip(self.layers, self.masks):
            if mask is not None:
                layer.weight.data = layer.weight.data * mask

    def optimize(self):
        self.apply_masks()  # N.B. This seems to be the correct order
        self.optimizer.step()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        x = self.layers[-1](x)  # output layer does not have an activation function
        return x

    def save_weights_and_bias(self, filename: str):
        print(f'Saving weights and bias to {filename}')
        data = {}
        for i, layer in enumerate(self.layers):
            data[f'W{i + 1}'] = layer.weight.data
            data[f'b{i + 1}'] = layer.bias.data
        save_dict_to_npz(filename, data)

    def load_weights_and_bias(self, filename: str):
        print(f'Loading weights and bias from {filename}')
        data = load_dict_from_npz(filename)
        for i, layer in enumerate(self.layers):
            layer.weight.data = data[f'W{i + 1}']
            layer.bias.data = data[f'b{i + 1}']

    def info(self):
        index = 1
        for layer in self.layers:
            if isinstance(layer, Linear):
                pp(f'W{index}', layer.weight.data)
                pp(f'b{index}', layer.bias.data)
                index += 1

    def __str__(self):
        def density_info(layer, mask: torch.Tensor):
            if mask is not None:
                n, N = torch.count_nonzero(mask), mask.numel()
            else:
                n, N = layer.weight.numel(), layer.weight.numel()
            return f'{n}/{N} ({100 * n / N:.8f}%)'

        density_info = [density_info(layer, mask) for layer, mask in zip(self.layers, self.masks)]
        return f'{super().__str__()}\nscheduler = {self.learning_rate}\nlayer densities: {", ".join(density_info)}\n'


class MLPPyTorchTRelu(MLPPyTorch):
    """ PyTorch Multilayer perceptron that supports sparse layers using binary masks.
        It uses a trimmed ReLU activation function.
    """
    def __init__(self, layer_sizes, layer_densities, epsilon: float):
        super().__init__(layer_sizes, layer_densities)
        self.epsilon = epsilon
        print(f'epsilon = {epsilon}')

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            z = self.layers[i](x)
            x = torch.where(z < self.epsilon, 0, z)  # apply trimmed ReLU to z
        x = self.layers[-1](x)  # output layer does not have an activation function
        return x


def parse_learning_rate(text: str) -> float:
    try:
        func = parse_function_call(text)
        if func.name == 'Constant':
            return func.as_float('lr')
        elif func.name == 'MultiStepLR':
            return func.as_float('lr')
    except:
        pass
    raise RuntimeError(f"Could not parse learning rate from '{text}'")


def parse_learning_rate_scheduler(text: str, optimizer) -> torch.optim.lr_scheduler:
    try:
        func = parse_function_call(text)
        if func.name == 'Constant':
            lr = func.as_float('lr')
            return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        elif func.name == 'MultiStepLR':
            lr = func.as_float('lr')
            milestones = [int(x) for x in func.as_string('milestones').split('|')]
            gamma = func.as_float('gamma')
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma, last_epoch=-1)
    except:
        pass
    raise RuntimeError(f"Could not parse learning rate scheduler '{text}'")


def parse_loss_function(text: str):
    if text == "SoftmaxCrossEntropy":
        return nn.CrossEntropyLoss()
    else:
        raise RuntimeError(f"unsupported loss function '{text}'")


# N.B. We only support one optimizer for all layers.
def parse_optimizer(text: str, M: MLPPyTorch, lr: float) -> optim.SGD:
    try:
        momentum = 0.0
        nesterov = False
        if text == 'GradientDescent':
            pass
        elif text.startswith('Momentum'):
            m = re.match(r'Momentum\((.*)\)', text)
            momentum = float(m.group(1))
        elif text.startswith('Nesterov'):
            m = re.match(r'Nesterov\((.*)\)', text)
            momentum = float(m.group(1))
            nesterov = True
        return optim.SGD(M.parameters(), lr=lr, momentum=momentum, nesterov=nesterov)
    except:
        pass
    raise RuntimeError(f"could not parse optimizer '{text}'")


def make_torch_model(args, linear_layer_sizes, densities):
    lr = parse_learning_rate(args.learning_rate)
    M = MLPPyTorch(linear_layer_sizes, densities) if args.trim_relu == 0 else MLPPyTorchTRelu(linear_layer_sizes, densities, args.trim_relu)
    M.optimizer = parse_optimizer(args.optimizers, M, lr)
    M.loss = parse_loss_function(args.loss)
    M.learning_rate = parse_learning_rate_scheduler(args.learning_rate, M.optimizer)
    for layer in M.layers:
        nn.init.xavier_uniform_(layer.weight)
    M.apply_masks()
    return M


def compute_accuracy_torch(M: MLPPyTorch, data_loader):
    N = len(data_loader.dataset)  # N is the number of examples
    total_correct = 0
    for X, T in data_loader:
        Y = M(X)
        predicted = Y.argmax(axis=1)  # the predicted classes for the batch
        total_correct += (predicted == T).sum().item()
    return total_correct / N


def compute_loss_torch(M: MLPPyTorch, data_loader):
    N = len(data_loader.dataset)  # N is the number of examples
    batch_size = N // len(data_loader)
    total_loss = 0.0
    for X, T in data_loader:
        Y = M(X)
        total_loss += M.loss(Y, T).sum()
    return batch_size * total_loss / N


def train_torch(M, train_loader, test_loader, epochs, debug=False):
    M.train()  # Set model in training mode
    watch = StopWatch()

    print_epoch(epoch=0,
                lr=M.optimizer.param_groups[0]["lr"],
                loss=compute_loss_torch(M, train_loader),
                train_accuracy=compute_accuracy_torch(M, train_loader),
                test_accuracy=compute_accuracy_torch(M, test_loader),
                elapsed=0)

    for epoch in range(epochs):
        elapsed = 0.0
        for k, (X, T) in enumerate(train_loader):
            watch.reset()
            M.optimizer.zero_grad()
            Y = M(X)

            if debug:
                Y.requires_grad = True
                Y.retain_grad()

            loss = M.loss(Y, T)
            loss.backward()

            if debug:
                print(f'epoch: {epoch} batch: {k}')
                M.info()
                DY = Y.grad.detach()
                pp("X", X.T)
                pp("Y", Y.T)
                pp("DY", DY.T)

            M.optimize()
            elapsed += watch.seconds()

        print_epoch(epoch=epoch + 1,
                    lr=M.optimizer.param_groups[0]["lr"],
                    loss=compute_loss_torch(M, train_loader),
                    train_accuracy=compute_accuracy_torch(M, train_loader),
                    test_accuracy=compute_accuracy_torch(M, test_loader),
                    elapsed=elapsed)

        M.learning_rate.step()  # N.B. this updates the learning rate in M.optimizer


# At every epoch a new dataset in .npz format is read from datadir.
def train_torch_preprocessed(M, datadir, epochs, batch_size, debug=False):
    M.train()  # Set model in training mode
    watch = StopWatch()

    train_loader, test_loader = create_npz_dataloaders(f'{datadir}/epoch0.npz', batch_size=batch_size)

    print_epoch(epoch=0,
                lr=M.optimizer.param_groups[0]["lr"],
                loss=compute_loss_torch(M, train_loader),
                train_accuracy=compute_accuracy_torch(M, train_loader),
                test_accuracy=compute_accuracy_torch(M, test_loader),
                elapsed=0)

    for epoch in range(epochs):
        if epoch > 0:
            train_loader, test_loader = create_npz_dataloaders(f'{datadir}/epoch{epoch}.npz', batch_size)

        elapsed = 0.0
        for k, (X, T) in enumerate(train_loader):
            if debug:
                X.requires_grad_(True)

            watch.reset()
            M.optimizer.zero_grad()
            Y = M(X)

            if debug:
                Y.retain_grad()

            loss = M.loss(Y, T)
            loss.backward()

            if debug:
                print(f'epoch: {epoch} batch: {k}')
                M.info()
                DY = Y.grad.detach()
                pp("X", X)
                pp("Y", Y)
                pp("DY", DY)

            M.optimize()
            elapsed += watch.seconds()

        print_epoch(epoch=epoch + 1,
                    lr=M.optimizer.param_groups[0]["lr"],
                    loss=compute_loss_torch(M, train_loader),
                    train_accuracy=compute_accuracy_torch(M, train_loader),
                    test_accuracy=compute_accuracy_torch(M, test_loader),
                    elapsed=elapsed)

        M.learning_rate.step()  # N.B. this updates the learning rate in M.optimizer


def make_argument_parser():
    cmdline_parser = argparse.ArgumentParser()

    # randomness
    cmdline_parser.add_argument("--seed", help="The initial seed of the random generator", type=int)

    # model parameters
    cmdline_parser.add_argument('--sizes', type=str, default='3072,128,64,10', help='A comma separated list of layer sizes, e.g. "3072,128,64,10".')
    cmdline_parser.add_argument('--densities', type=str, help='A comma separated list of layer densities, e.g. "0.05,0.05,1.0".')
    cmdline_parser.add_argument('--overall-density', type=float, default=1.0, help='The overall density of the layers.')
    cmdline_parser.add_argument('--trim-relu', type=float, default=0.0, help='The threshold for trimming ReLU outputs.')
    cmdline_parser.add_argument('--layers', type=str, help='A semi-colon separated lists of layers.')

    # optimizer
    cmdline_parser.add_argument('--momentum', type=float, default=0.9, help='the momentum value (default: off)')
    cmdline_parser.add_argument("--nesterov", help="apply nesterov", action="store_true")

    # learning rate
    cmdline_parser.add_argument("--learning-rate", type=str, help="The learning rate scheduler")

    # loss function
    cmdline_parser.add_argument('--loss', type=str, help='The loss function')

    # training
    cmdline_parser.add_argument("--epochs", help="The number of epochs", type=int, default=100)
    cmdline_parser.add_argument("--batch-size", help="The batch size", type=int, default=1)

    # optimizer
    cmdline_parser.add_argument("--optimizers", type=str, help="The optimizer (GradientDescent, Momentum(<mu>), Nesterov(<mu>))", default="GradientDescent")

    # dataset
    cmdline_parser.add_argument('--datadir', type=str, default='', help='the data directory (default: ./data)')
    cmdline_parser.add_argument("--augmented", help="use data loaders with augmentation", action="store_true")
    cmdline_parser.add_argument("--preprocessed", help="folder with preprocessed datasets for each epoch")

    # load/save weights
    cmdline_parser.add_argument('--init-weights', type=str, default='None', help='The initial weights for the layers')
    cmdline_parser.add_argument('--save-weights', type=str, help='Save weights and bias to a file in .npz format')
    cmdline_parser.add_argument('--load-weights', type=str, help='Load weights and bias from a file in .npz format')

    # print options
    cmdline_parser.add_argument("--precision", help="The precision used for printing matrices", type=int, default=8)
    cmdline_parser.add_argument("--edgeitems", help="The edgeitems used for printing matrices", type=int, default=3)
    cmdline_parser.add_argument("--debug", help="print debug information", action="store_true")
    cmdline_parser.add_argument("--info", help="print information about the MLP", action="store_true")

    # multi-threading
    cmdline_parser.add_argument("--threads", help="The number of threads being used", type=int)

    # timer
    cmdline_parser.add_argument("--timer", help="Enable timer messages", action="store_true")

    return cmdline_parser


def check_command_line_arguments(args):
    if args.augmented and args.preprocessed:
        raise RuntimeError('the combination of --augmented and --preprocessed is unsupported')

    if args.densities and args.overall_density:
        raise RuntimeError('the options --densities and --overall-density cannot be used simultaneously')

    if not args.datadir and not args.preprocessed:
        raise RuntimeError('at least one of the options --datadir and --preprocessed must be set')


def print_command_line_arguments(args):
    print("python3 " + " ".join(shlex.quote(arg) if " " in arg else arg for arg in sys.argv) + '\n')


def initialize_frameworks(args):
    if args.seed:
        torch.manual_seed(args.seed)

    torch.set_printoptions(precision=args.precision, edgeitems=args.edgeitems, threshold=5, sci_mode=False, linewidth=160)

    # avoid 'Too many open files' error when using data loaders
    torch.multiprocessing.set_sharing_strategy('file_system')


def main():
    cmdline_parser = make_argument_parser()
    args = cmdline_parser.parse_args()
    check_command_line_arguments(args)
    print_command_line_arguments(args)

    initialize_frameworks(args)

    if args.datadir:
        if args.augmented:
            train_loader, test_loader = create_cifar10_augmented_dataloaders(args.batch_size, args.batch_size, args.datadir)
        else:
            train_loader, test_loader = create_cifar10_dataloaders(args.batch_size, args.batch_size, args.datadir)
    else:
        train_loader, test_loader = None, None

    linear_layer_sizes = [int(s) for s in args.sizes.split(',')]

    if args.densities:
        linear_layer_densities = list(float(d) for d in args.densities.split(','))
    elif args.overall_density:
        linear_layer_densities = compute_sparse_layer_densities(args.overall_density, linear_layer_sizes)
    else:
        linear_layer_densities = [1.0] * (len(linear_layer_sizes) - 1)

    M = make_torch_model(args, linear_layer_sizes, linear_layer_densities)
    print('=== PyTorch model ===')
    print(M)

    if args.load_weights:
        M.load_weights_and_bias(args.load_weights)
    if args.save_weights:
        M.save_weights_and_bias(args.save_weights)

    if args.epochs > 0:
        print('\n=== Training PyTorch model ===')
        if args.preprocessed:
            train_torch_preprocessed(M, args.preprocessed, args.epochs, args.batch_size, args.debug)
        else:
            train_torch(M, train_loader, test_loader, args.epochs, args.debug)


if __name__ == '__main__':
    main()
