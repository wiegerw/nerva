#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import argparse
import os
import tempfile
from timeit import default_timer as timer
from typing import List, Union
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import nerva.dataset
import nerva.layers
import nerva.learning_rate
import nerva.loss
import nerva.optimizers
from sparselearning.core import Masking


def to_eigen(x: np.ndarray):
    return x.reshape(x.shape[1], x.shape[0], order='F').T


def from_eigen(x: np.ndarray):
    return x.reshape(x.shape[1], x.shape[0], order='C').T


# Loads a number of Numpy arrays from an .npy file and returns them in a list
def load_numpy_arrays_from_npy_file(filename: str) -> List[np.ndarray]:
    arrays = []
    try:
        with open(filename, "rb") as f:
            while True:
                arrays.append(from_eigen(np.load(f, allow_pickle=True)))
    except IOError:
        pass
    return arrays


def flatten_torch(x: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    return x.reshape(shape[0], -1)


def flatten_numpy(x: np.ndarray) -> np.ndarray:
    shape = x.shape
    return x.reshape(shape[0], -1)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return np.asfortranarray(x.detach().numpy().T)


def to_one_hot_torch(x: torch.Tensor) -> np.ndarray:
    return to_numpy(torch.nn.functional.one_hot(x, num_classes = 10).float())


def to_one_hot_numpy(x: np.ndarray, n_classes: int):
    return flatten_numpy(np.asfortranarray(np.eye(n_classes)[x].T))


def create_cifar10_dataloaders(batch_size, test_batch_size, datadir='./data'):
    """Creates augmented train and test data loaders."""

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
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size,
        num_workers=8,
        pin_memory=True, shuffle=True)

    print('Train loader length', len(train_loader))

    test_dataset = datasets.CIFAR10(datadir, False, test_transform, download=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        test_batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True)

    return train_loader, test_loader


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


def normalize_cifar_data(X: np.array, mean=None, std=None):
    if not mean:
        mean = X.mean(axis=(0, 1, 2))
    if not std:
        std = X.std(axis=(0, 1, 2))
    return (X  - mean) / std


def flatten_cifar_data(X: np.array):
    shape = X.shape
    return X.reshape(shape[0], -1)


def load_cifar10_data(datadir):
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    trainset = torchvision.datasets.CIFAR10(root=datadir, train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root=datadir, train=False, download=True)

    Xtrain = normalize_cifar_data(trainset.data / 255, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    Xtrain = flatten_cifar_data(Xtrain)
    Xtrain = torch.Tensor(Xtrain)
    Ttrain = torch.LongTensor(trainset.targets)

    Xtest = normalize_cifar_data(testset.data / 255)
    Xtest = flatten_cifar_data(Xtest)
    Xtest = torch.Tensor(Xtest)
    Ttest = torch.LongTensor(testset.targets)

    return Xtrain, Ttrain, Xtest, Ttest


def compute_accuracy1(M: nn.Module, data_loader):
    N = len(data_loader.dataset)  # N is the number of examples
    total_correct = 0
    for X, T in data_loader:
        Y = M(X)
        predicted = Y.argmax(axis=1)  # the predicted classes for the batch
        total_correct += (predicted == T).sum().item()
    return total_correct / N


def compute_loss1(M: nn.Module, loss, data_loader):
    N = len(data_loader.dataset)  # N is the number of examples
    batch_size = N // len(data_loader)
    total_loss = 0.0
    for X, T in data_loader:
        Y = M(X)
        total_loss += loss(Y, T).sum()
    return batch_size * total_loss / N


def compute_accuracy2(M: nerva.layers.Sequential, data_loader):
    N = len(data_loader.dataset)  # N is the number of examples
    total_correct = 0
    for X, T in data_loader:
        X = to_numpy(X)
        T = to_numpy(T)
        Y = M.feedforward(X)
        predicted = Y.argmax(axis=0)  # the predicted classes for the batch
        total_correct += (predicted == T).sum().item()
    return total_correct / N


def compute_loss2(M: nerva.layers.Sequential, loss, data_loader):
    N = len(data_loader.dataset)  # N is the number of examples
    total_loss = 0.0
    for X, T in data_loader:
        X = to_numpy(X)
        T = to_one_hot_numpy(T, 10)
        Y = M.feedforward(X)
        total_loss += loss.value(Y, T)
    return total_loss / N


class MLP1(nn.Module):
    """ Multi-Layer Perceptron """
    def __init__(self, sizes):
        super().__init__()
        self.optimizer = None
        self.mask = None
        n = len(sizes) - 1  # the number of layers
        self.layers = nn.ModuleList()
        for i in range(n):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        x = self.layers[-1](x)  # output layer does not have an activation function
        return x

    def set_mask(self, mask, optimizer, density, sparse_init='ER'):
        self.mask = mask
        self.optimizer = optimizer
        if mask:
            mask.add_module(self, sparse_init=sparse_init, density=density)

    def optimize(self):
        if self.mask is not None:
            self.mask.step()
        else:
            self.optimizer.step()

    def weights(self) -> List[torch.Tensor]:
        return [layer.weight for layer in self.layers]

    def bias(self) -> List[torch.Tensor]:
        return [layer.bias for layer in self.layers]

    def export_weights(self, filename: str):
        with open(filename, "wb") as f:
            for layer in self.layers:
                np.save(f, to_eigen(layer.weight.detach().numpy()))

    def export_bias(self, filename: str):
        with open(filename, "wb") as f:
            for layer in self.layers:
                np.save(f, layer.bias.detach().numpy())


class MLP2(nerva.layers.Sequential):
    def __init__(self, sizes, densities, optimizer, batch_size):
        super().__init__()
        n = len(densities)  # the number of layers
        activations = [nerva.layers.ReLU()] * (n-1) + [nerva.layers.NoActivation()]
        output_sizes = sizes[1:]
        for (density, size, activation) in zip(densities, output_sizes, activations):
            if density == 1.0:
                 self.add(nerva.layers.Dense(size, activation=activation, optimizer=optimizer))
            else:
                self.add(nerva.layers.Sparse(size, 1.0 - density, activation=activation, optimizer=optimizer))
        self.compile(sizes[0], batch_size)

    def weights(self) -> List[torch.Tensor]:
        filename = tempfile.NamedTemporaryFile().name + '_weights.npy'
        self.export_weights(filename)
        weights = load_numpy_arrays_from_npy_file(filename)
        return [torch.Tensor(W) for W in weights]

    def bias(self) -> List[torch.Tensor]:
        filename = tempfile.NamedTemporaryFile().name + '_bias.npy'
        self.export_bias(filename)
        bias = load_numpy_arrays_from_npy_file(filename)
        return [torch.Tensor(b) for b in bias]


# Copies models and weights from model1 to model2
def copy_weights_and_biases(model1: nn.Module, model2: nerva.layers.Sequential):
    name = tempfile.NamedTemporaryFile().name
    filename1 = name + '_weights.npy'
    filename2 = name + '_bias.npy'
    model1.export_weights(filename1)
    model2.import_weights(filename1)
    model1.export_bias(filename2)
    model2.import_bias(filename2)


def pp(name: str, x: Union[torch.Tensor, np.ndarray]):
    if isinstance(x, np.ndarray):
        x = torch.Tensor(x.T)
    if len(x.shape) == 1:
        print(f'{name} ({x.shape[0]})\n{x.data}')
    else:
        print(f'{name} ({x.shape[0]}x{x.shape[1]})\n{x.data}')


def print_model_info(M):
    W = M.weights()
    b = M.bias()
    for i in range(len(W)):
        pp(f'W{i+1}', W[i])
        pp(f'b{i+1}', b[i])


def train_pytorch(M, train_loader, test_loader, optimizer, criterion, learning_rate, epochs, show: bool):
    for epoch in range(epochs):
        start = timer()
        for k, (X, T) in enumerate(train_loader):
            optimizer.zero_grad()
            Y = M(X)
            Y.retain_grad()
            loss = criterion(Y, T)
            loss.backward()
            M.optimize()
            elapsed = timer() - start

            if show:
                print(f'epoch: {epoch} batch: {k}')
                pp('Y', Y)
                pp('DY', Y.grad.detach())

        print(f'epoch {epoch + 1:3}  '
              f'lr: {optimizer.param_groups[0]["lr"]:.4f}  '
              f'loss: {compute_loss1(M, criterion, train_loader):.3f}  '
              f'train accuracy: {compute_accuracy1(M, train_loader):.3f}  '
              f'test accuracy: {compute_accuracy1(M, test_loader):.3f}  '
              f'time: {elapsed:.3f}'
             )

        learning_rate.step()  # N.B. this updates the learning rate in optimizer


def train_nerva(M, train_loader, test_loader, criterion, learning_rate, epochs, batch_size, show: bool):
    for epoch in range(epochs):
        start = timer()
        lr = learning_rate(epoch)
        for k, (X, T) in enumerate(train_loader):
            X = to_numpy(X)
            T = to_one_hot_numpy(T, 10)
            Y = M.feedforward(X)
            DY = criterion.gradient(Y, T) / batch_size
            M.backpropagate(Y, DY)
            M.optimize(lr)
            elapsed = timer() - start

            if show:
                print(f'epoch: {epoch} batch: {k}')
                pp('Y', Y)
                pp('DY', DY)

        print(f'epoch {epoch + 1:3}  '
              f'lr: {lr:.4f}  '
              f'loss: {compute_loss2(M, criterion, train_loader):.3f}  '
              f'train accuracy: {compute_accuracy2(M, train_loader):.3f}  '
              f'test accuracy: {compute_accuracy2(M, test_loader):.3f}  '
              f'time: {elapsed:.3f}'
             )


def make_nerva_optimizer(momentum=0.0, nesterov=False) -> nerva.optimizers.Optimizer:
    if nesterov:
        return nerva.optimizers.Nesterov(momentum)
    elif momentum > 0.0:
        return nerva.optimizers.Momentum(momentum)
    else:
        return nerva.optimizers.GradientDescent()


def compute_densities(density: float, sizes: List[int], erk_power_scale: float = 1.0) -> List[float]:
    layer_shapes = [(sizes[i], sizes[i+1]) for i in range(len(sizes) - 1)]
    n = len(layer_shapes)  # the number of layers

    if density == 1.0:
        return [1.0] * n

    total_params = sum(rows * columns for (rows, columns) in layer_shapes)

    dense_layers = set()
    while True:
        divisor = 0
        rhs = 0
        raw_probabilities = [0.0] * n
        for i, (rows, columns) in enumerate(layer_shapes):
            n_param = rows * columns
            n_zeros = n_param * (1 - density)
            n_ones = n_param * density
            if i in dense_layers:
                rhs -= n_zeros
            else:
                rhs += n_ones
                raw_probabilities[i] = ((rows + columns) / (rows * columns)) ** erk_power_scale
                divisor += raw_probabilities[i] * n_param
        epsilon = rhs / divisor
        max_prob = max(raw_probabilities)
        max_prob_one = max_prob * epsilon
        if max_prob_one > 1:
            for j, mask_raw_prob in enumerate(raw_probabilities):
                if mask_raw_prob == max_prob:
                    print(f"Sparsity of layer:{j} had to be set to 0.")
                    dense_layers.add(j)
        else:
            break

    # Compute the densities
    densities = [0.0] * n
    total_nonzero = 0.0
    for i, (rows, columns) in enumerate(layer_shapes):
        n_param = rows * columns
        if i in dense_layers:
            densities[i] = 1.0
        else:
            probability_one = epsilon * raw_probabilities[i]
            densities[i] = probability_one
        print(f"layer: {i}, shape: {(rows,columns)}, density: {densities[i]}")
        total_nonzero += densities[i] * n_param
    print(f"Overall sparsity {total_nonzero / total_params:.4f}")
    return densities


def make_mask(model: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              density,
              train_loader,
              sparse_init='ER',
             ) -> Masking:
    mask = Masking(optimizer,
                   prune_rate=0.5,
                   prune_mode='none',
                   prune_rate_decay=None,
                   prune_interval=0,
                   growth_mode='random',
                   redistribution_mode='none',
                   train_loader=train_loader,
                   )
    mask.add_module(model, sparse_init=sparse_init, density=density)
    return mask


def main():
    cmdline_parser = argparse.ArgumentParser()
    cmdline_parser.add_argument("--show", help="Show data and intermediate results", action="store_true")
    cmdline_parser.add_argument("--batch-size", help="The batch size", type=int, default=1)
    cmdline_parser.add_argument("--seed", help="The initial seed of the random generator", type=int)
    cmdline_parser.add_argument("--precision", help="The precision used for printing", type=int, default=4)
    cmdline_parser.add_argument("--edgeitems", help="The edgeitems used for printing matrices", type=int, default=3)
    cmdline_parser.add_argument("--epochs", help="The number of epochs", type=int, default=100)
    cmdline_parser.add_argument("--lr", help="The learning rate", type=float, default=0.1)
    cmdline_parser.add_argument('--momentum', type=float, default=0.9, help='the momentum value (default: off)')
    cmdline_parser.add_argument("--nesterov", help="apply nesterov", action="store_true")
    cmdline_parser.add_argument('--datadir', type=str, default='./data', help='the data directory (default: ./data)')
    cmdline_parser.add_argument("--augmented", help="use data loaders with augmentation", action="store_true")
    cmdline_parser.add_argument('--density', type=float, default=1.0, help='The density of the overall sparse network.')
    cmdline_parser.add_argument('--sizes', type=str, default='3072,128,64,10', help='A comma separated list of layer sizes, e.g. "3072,128,64,10".')
    cmdline_parser.add_argument("--copy", help="copy weights and biases from the PyTorch model to the Nerva model", action="store_true")
    cmdline_parser.add_argument("--nerva", help="Train using a Nerva model", action="store_true")
    cmdline_parser.add_argument("--torch", help="Train using a PyTorch model", action="store_true")
    cmdline_parser.add_argument("--info", help="Print detailed info about the models", action="store_true")
    args = cmdline_parser.parse_args()

    print('=== Command line arguments ===')
    print(args)

    if args.seed:
        torch.manual_seed(args.seed)

    torch.set_printoptions(precision=args.precision, edgeitems=args.edgeitems, threshold=5, sci_mode=False, linewidth=120)

    # avoid 'Too many open files' error when using data loaders
    torch.multiprocessing.set_sharing_strategy('file_system')

    if args.augmented:
        train_loader, test_loader = create_cifar10_dataloaders(args.batch_size, args.batch_size, args.datadir)
    else:
        Xtrain, Ttrain, Xtest, Ttest = load_cifar10_data(args.datadir)
        train_loader = TorchDataLoader(Xtrain, Ttrain, args.batch_size)
        test_loader = TorchDataLoader(Xtest, Ttest, args.batch_size)

    sizes = [int(s) for s in args.sizes.split(',')]
    densities = compute_densities(args.density, sizes)

    # parameters for the learning rate scheduler
    milestones = [int(args.epochs / 2), int(args.epochs * 3 / 4)]

    # create PyTorch model
    M1 = MLP1(sizes)
    loss1 = nn.CrossEntropyLoss()
    optimizer1 = optim.SGD(M1.parameters(), lr=args.lr, momentum=args.momentum, nesterov=args.nesterov)
    mask = make_mask(M1, optimizer1, args.density, train_loader) if args.density != 1.0 else None
    M1.set_mask(mask, optimizer1, args.density)  # this cannot be done during construction due to circular dependencies in PyTorch
    learning_rate1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, milestones=milestones, last_epoch=-1)
    print('\n=== PyTorch model ===')
    print(M1)
    print(loss1)
    print(learning_rate1)

    # create Nerva model
    optimizer2 = make_nerva_optimizer(args.momentum, args.nesterov)
    M2 = MLP2(sizes, densities, optimizer2, args.batch_size)
    loss2 = nerva.loss.SoftmaxCrossEntropyLoss()
    learning_rate2 = nerva.learning_rate.MultiStepLRScheduler(args.lr, milestones, 0.1)
    print('\n=== Nerva model ===')
    print(M2)
    print(loss2)
    print(learning_rate2)

    if args.copy:
        copy_weights_and_biases(M1, M2)

    if args.info:
        print('\n=== PyTorch info ===')
        print_model_info(M1)
        print('\n=== Nerva info ===')
        print_model_info(M2)

    if args.torch:
        print('\n=== Training PyTorch model ===')
        train_pytorch(M1, train_loader, test_loader, optimizer1, loss1, learning_rate1, args.epochs, args.show)
        print(f'Accuracy of the network on the 10000 test images: {100 * compute_accuracy1(M1, test_loader):.3f} %')

    if args.nerva:
        print('\n=== Training Nerva model ===')
        train_nerva(M2, train_loader, test_loader, loss2, learning_rate2, args.epochs, args.batch_size, args.show)
        print(f'Accuracy of the network on the 10000 test images: {100 * compute_accuracy2(M2, test_loader):.3f} %')


if __name__ == '__main__':
    main()
