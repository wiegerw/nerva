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


class MLP1(nn.Module):
    """ Multi-Layer Perceptron """
    def __init__(self, sizes):
        super().__init__()
        self.sizes = sizes
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

    def optimize(self):
        if self.mask is not None:
            self.mask.step()
        else:
            self.optimizer.step()

    def weights(self) -> List[torch.Tensor]:
        return [layer.weight.detach().numpy() for layer in self.layers]

    def bias(self) -> List[torch.Tensor]:
        return [layer.bias.detach().numpy() for layer in self.layers]

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
        self.sizes = sizes
        self.loss = None
        self.learning_rate = None
        n = len(densities)  # the number of layers
        activations = [nerva.layers.ReLU()] * (n-1) + [nerva.layers.NoActivation()]
        output_sizes = sizes[1:]
        for (density, size, activation) in zip(densities, output_sizes, activations):
            if density == 1.0:
                 self.add(nerva.layers.Dense(size, activation=activation, optimizer=optimizer))
            else:
                self.add(nerva.layers.Sparse(size, 1.0 - density, activation=activation, optimizer=optimizer))
        self.compile(sizes[0], batch_size)

    def weights(self) -> List[np.ndarray]:
        filename = tempfile.NamedTemporaryFile().name + '_weights.npy'
        self.export_weights(filename)
        return load_numpy_arrays_from_npy_file(filename)

    def bias(self) -> List[np.ndarray]:
        def flatten(x: np.ndarray):
            if len(x.shape) == 2 and x.shape[1] == 1:
                return x.reshape(x.shape[0])
            else:
                return x

        filename = tempfile.NamedTemporaryFile().name + '_bias.npy'
        self.export_bias(filename)
        bias = load_numpy_arrays_from_npy_file(filename)
        # N.B. The shape of the bias can be (128,1), in which case we flatten it to (128).
        return [flatten(b) for b in bias]


def compute_accuracy1(M: MLP1, data_loader):
    N = len(data_loader.dataset)  # N is the number of examples
    total_correct = 0
    for X, T in data_loader:
        Y = M(X)
        predicted = Y.argmax(axis=1)  # the predicted classes for the batch
        total_correct += (predicted == T).sum().item()
    return total_correct / N


def compute_loss1(M: MLP1, data_loader):
    N = len(data_loader.dataset)  # N is the number of examples
    batch_size = N // len(data_loader)
    total_loss = 0.0
    for X, T in data_loader:
        Y = M(X)
        total_loss += M.loss(Y, T).sum()
    return batch_size * total_loss / N


def compute_accuracy2(M: MLP2, data_loader):
    N = len(data_loader.dataset)  # N is the number of examples
    total_correct = 0
    for X, T in data_loader:
        X = to_numpy(X)
        T = to_numpy(T)
        Y = M.feedforward(X)
        predicted = Y.argmax(axis=0)  # the predicted classes for the batch
        total_correct += (predicted == T).sum().item()
    return total_correct / N


def compute_loss2(M: MLP2, data_loader):
    N = len(data_loader.dataset)  # N is the number of examples
    total_loss = 0.0
    for X, T in data_loader:
        X = to_numpy(X)
        T = to_one_hot_numpy(T, 10)
        Y = M.feedforward(X)
        total_loss += M.loss.value(Y, T)
    return total_loss / N


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


def l1_norm(x: np.ndarray):
    return np.abs(x).sum()


def l2_norm(x: np.ndarray):
    return np.linalg.norm(x)


def compute_weight_difference(M1, M2):
    wdiff = [l1_norm(W1 - W2) for W1, W2 in zip(M1.weights(), M2.weights())]
    bdiff = [l1_norm(b1 - b2) for b1, b2 in zip(M1.bias(), M2.bias())]
    print(f'weight differences: {wdiff} bias differences: {bdiff}')


def compute_matrix_difference(name, X1: np.ndarray, X2: np.ndarray):
    print(f'{name} difference: {l1_norm(X1 - X2)}')


def train_pytorch(M, train_loader, test_loader, epochs, show: bool):
    for epoch in range(epochs):
        start = timer()
        for k, (X, T) in enumerate(train_loader):
            M.optimizer.zero_grad()
            Y = M(X)
            Y.retain_grad()
            loss = M.loss(Y, T)
            loss.backward()
            M.optimize()
            elapsed = timer() - start

            if show:
                print(f'epoch: {epoch} batch: {k}')
                pp('Y', Y)
                pp('DY', Y.grad.detach())

        print(f'epoch {epoch + 1:3}  '
              f'lr: {M.optimizer.param_groups[0]["lr"]:.4f}  '
              f'loss: {compute_loss1(M, train_loader):.3f}  '
              f'train accuracy: {compute_accuracy1(M, train_loader):.3f}  '
              f'test accuracy: {compute_accuracy1(M, test_loader):.3f}  '
              f'time: {elapsed:.3f}'
             )

        M.learning_rate.step()  # N.B. this updates the learning rate in M.optimizer


def train_nerva(M, train_loader, test_loader, epochs, show: bool):
    n_classes = M.sizes[-1]
    batch_size = len(train_loader.dataset) // len(train_loader)
    for epoch in range(epochs):
        start = timer()
        lr = M.learning_rate(epoch)
        for k, (X, T) in enumerate(train_loader):
            X = to_numpy(X)
            T = to_one_hot_numpy(T, n_classes)
            Y = M.feedforward(X)
            DY = M.loss.gradient(Y, T) / batch_size
            M.backpropagate(Y, DY)
            M.optimize(lr)
            elapsed = timer() - start

            if show:
                print(f'epoch: {epoch} batch: {k}')
                pp('Y', Y)
                pp('DY', DY)

        print(f'epoch {epoch + 1:3}  '
              f'lr: {lr:.4f}  '
              f'loss: {compute_loss2(M, train_loader):.3f}  '
              f'train accuracy: {compute_accuracy2(M, train_loader):.3f}  '
              f'test accuracy: {compute_accuracy2(M, test_loader):.3f}  '
              f'time: {elapsed:.3f}'
             )


def train_both(M1: MLP1, M2: MLP2, train_loader, test_loader, epochs, show: bool):
    n_classes = M2.sizes[-1]
    batch_size = len(train_loader.dataset) // len(train_loader)

    if show:
        compute_weight_difference(M1, M2)

    for epoch in range(epochs):
        start = timer()
        lr = M2.learning_rate(epoch)

        for k, (X1, T1) in enumerate(train_loader):
            M1.optimizer.zero_grad()
            Y1 = M1(X1)
            Y1.retain_grad()
            loss = M1.loss(Y1, T1)
            loss.backward()
            M1.optimize()

            # if show:
            #     print(f'epoch: {epoch} batch: {k}')
            #     pp('Y', Y1)
            #     pp('DY', Y1.grad.detach())

            X2 = to_numpy(X1)
            T2 = to_one_hot_numpy(T1, n_classes)
            Y2 = M2.feedforward(X2)
            DY2 = M2.loss.gradient(Y2, T2) / batch_size
            M2.backpropagate(Y2, DY2)
            M2.optimize(lr)

            # if show:
            #     print(f'epoch: {epoch} batch: {k}')
            #     pp('Y', Y2)
            #     pp('DY', DY2)

            if show:
                print(f'epoch: {epoch} batch: {k}')
                compute_matrix_difference('Y', Y1.detach().numpy().T, Y2)
                compute_matrix_difference('DY', Y1.grad.detach().numpy().T, DY2)
                compute_weight_difference(M1, M2)

            elapsed = timer() - start

        print(f'epoch {epoch + 1:3}  '
              f'lr: {M1.optimizer.param_groups[0]["lr"]:.4f}  '
              f'loss: {compute_loss1(M1, train_loader):.3f}  '
              f'train accuracy: {compute_accuracy1(M1, train_loader):.3f}  '
              f'test accuracy: {compute_accuracy1(M1, test_loader):.3f}  '
              f'time: {elapsed:.3f}'
             )

        print(f'epoch {epoch + 1:3}  '
              f'lr: {lr:.4f}  '
              f'loss: {compute_loss2(M2, train_loader):.3f}  '
              f'train accuracy: {compute_accuracy2(M2, train_loader):.3f}  '
              f'test accuracy: {compute_accuracy2(M2, test_loader):.3f}  '
              f'time: {elapsed:.3f}'
             )

        M1.learning_rate.step()


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
              density,
              train_loader,
              sparse_init='ER',
             ) -> Masking:
    mask = Masking(model.optimizer,
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
    M1.loss = nn.CrossEntropyLoss()
    M1.optimizer = optim.SGD(M1.parameters(), lr=args.lr, momentum=args.momentum, nesterov=args.nesterov)
    if args.density != None:
        mask = make_mask(M1, args.density, train_loader)
        mask.add_module(M1, density=args.density, sparse_init='ER')
        M1.mask = mask
    M1.learning_rate = torch.optim.lr_scheduler.MultiStepLR(M1.optimizer, milestones=milestones, last_epoch=-1)
    print('\n=== PyTorch model ===')
    print(M1)
    print(M1.loss)
    print(M1.learning_rate)

    # create Nerva model
    optimizer2 = make_nerva_optimizer(args.momentum, args.nesterov)
    M2 = MLP2(sizes, densities, optimizer2, args.batch_size)
    M2.loss = nerva.loss.SoftmaxCrossEntropyLoss()
    M2.learning_rate = nerva.learning_rate.MultiStepLRScheduler(args.lr, milestones, 0.1)
    print('\n=== Nerva model ===')
    print(M2)
    print(M2.loss)
    print(M2.learning_rate)

    if args.copy:
        copy_weights_and_biases(M1, M2)

    if args.info:
        print('\n=== PyTorch info ===')
        print_model_info(M1)
        print('\n=== Nerva info ===')
        print_model_info(M2)

    if args.torch and args.nerva:
        print('\n=== Training PyTorch and Nerva model ===')
        train_both(M1, M2, train_loader, test_loader, args.epochs, args.show)
        print(f'Accuracy of the network M1 on the 10000 test images: {100 * compute_accuracy1(M1, test_loader):.3f} %')
        print(f'Accuracy of the network M2 on the 10000 test images: {100 * compute_accuracy2(M2, test_loader):.3f} %')
    elif args.torch:
        print('\n=== Training PyTorch model ===')
        train_pytorch(M1, train_loader, test_loader, args.epochs, args.show)
        print(f'Accuracy of the network on the 10000 test images: {100 * compute_accuracy1(M1, test_loader):.3f} %')
    elif args.nerva:
        print('\n=== Training Nerva model ===')
        train_nerva(M2, train_loader, test_loader, args.epochs, args.show)
        print(f'Accuracy of the network on the 10000 test images: {100 * compute_accuracy2(M2, test_loader):.3f} %')


if __name__ == '__main__':
    main()
