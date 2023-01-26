#!/usr/bin/env python3

import argparse
import os
import tempfile
from timeit import default_timer as timer
from typing import List, Tuple, Union
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
        n = len(sizes) - 1  # the number of layers
        self.layers = nn.ModuleList()
        for i in range(n):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        x = self.layers[-1](x)  # output layer does not have an activation function
        return x

    def export_weights(self, filename: str):
        with open(filename, "wb") as f:
            for layer in self.layers:
                np.save(f, np.asfortranarray(layer.weight.detach().numpy()))

    def export_bias(self, filename: str):
        with open(filename, "wb") as f:
            for layer in self.layers:
                np.save(f, np.asfortranarray(layer.bias.detach().numpy()))


class MLP2(nerva.layers.Sequential):
    def __init__(self, sizes, optimizer, batch_size):
        super().__init__()
        input_size = sizes[0]
        hidden_layer_sizes = sizes[1:-1]
        output_size = sizes[-1]
        for size in hidden_layer_sizes:
            self.add(nerva.layers.Dense(size, activation=nerva.layers.ReLU(), optimizer=optimizer))
        self.add(nerva.layers.Dense(output_size, activation=nerva.layers.NoActivation(), optimizer=optimizer))
        self.compile(input_size, batch_size)


# Copies models and weights from model1 to model2
def copy_weights_and_biases(model1: nn.Module, model2: nerva.layers.Sequential):
    name = tempfile.NamedTemporaryFile().name
    filename1 =  name + '_weights.npy'
    filename2 =  name + '_bias.npy'
    print('saving weights to', filename1)
    print('saving bias to', filename2)
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


def train_pytorch(M, train_loader, test_loader, optimizer, criterion, epochs, show: bool):
    print('Training...')
    for epoch in range(epochs):
        start = timer()
        for k, (X, T) in enumerate(train_loader):
            optimizer.zero_grad()
            Y = M(X)
            Y.retain_grad()
            loss = criterion(Y, T)
            loss.backward()

            if show:
                print(f'epoch: {epoch} batch: {k}')
                #pp('X', X)
                #pp('Y', Y)
                pp('DY', Y.grad.detach())

            optimizer.step()
            elapsed = timer() - start

        print(f'epoch {epoch + 1:3}  '
              f'loss: {compute_loss1(M, criterion, train_loader):.3f}  '
              f'train accuracy: {compute_accuracy1(M, train_loader):.3f}  '
              f'test accuracy: {compute_accuracy1(M, test_loader):.3f}  '
              f'time: {elapsed:.3f}'
             )


def train_nerva(M, train_loader, test_loader, criterion, epochs, batch_size, lr, show: bool):
    print('Training...')
    for epoch in range(epochs):  # loop over the dataset multiple times
        start = timer()
        for k, (X, T) in enumerate(train_loader):
            X = to_numpy(X)
            T = to_one_hot_numpy(T, 10)
            Y = M.feedforward(X)
            DY = criterion.gradient(Y, T) / batch_size
            M.backpropagate(Y, DY)
            M.optimize(lr)

            if show:
                print(f'epoch: {epoch} batch: {k}')
                #pp('X', X)
                #pp('Y', Y)
                pp('DY', DY)

            elapsed = timer() - start

        print(f'epoch {epoch + 1:3}  '
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


def make_models(sizes, densities, nerva_optimizer, batch_size):
    if not densities:
        M1 = MLP1(sizes)
        M2 = MLP2(sizes, nerva_optimizer, batch_size)
        return M1, M2
    raise RuntimeError('sparse layers are not supported yet')


def main():
    cmdline_parser = argparse.ArgumentParser()
    cmdline_parser.add_argument("--show", help="Show data and intermediate results", action="store_true")
    cmdline_parser.add_argument("--batch-size", help="The batch size", type=int, default=1)
    cmdline_parser.add_argument("--seed", help="The initial seed of the random generator", type=int)
    cmdline_parser.add_argument("--precision", help="The precision used for printing", type=int, default=4)
    cmdline_parser.add_argument("--edgeitems", help="The edgeitems used for printing matrices", type=int, default=3)
    cmdline_parser.add_argument("--epochs", help="The number of epochs", type=int, default=100)
    cmdline_parser.add_argument("--learning-rate", help="The learning rate", type=float, default=0.001)
    cmdline_parser.add_argument("--run", help="The frameworks to run (both, nerva, pytorch)", type=str, default='both')
    cmdline_parser.add_argument('--momentum', type=float, default=0.9, help='the momentum value (default: off)')
    cmdline_parser.add_argument("--nesterov", help="apply nesterov", action="store_true")
    cmdline_parser.add_argument('--datadir', type=str, default='./data', help='the data directory (default: ./data)')
    cmdline_parser.add_argument("--augmented", help="use data loaders with augmentation", action="store_true")
    cmdline_parser.add_argument('--density', type=float, default=0.05, help='The density of the overall sparse network.')
    args = cmdline_parser.parse_args()

    if args.seed:
        torch.manual_seed(args.seed)

    torch.set_printoptions(precision=args.precision, edgeitems=args.edgeitems, threshold=5)
    if args.augmented:
        train_loader, test_loader = create_cifar10_dataloaders(args.batch_size, args.batch_size, args.datadir)
    else:
        Xtrain, Ttrain, Xtest, Ttest = load_cifar10_data(args.datadir)
        train_loader = TorchDataLoader(Xtrain, Ttrain, args.batch_size)
        test_loader = TorchDataLoader(Xtest, Ttest, args.batch_size)

    sizes = [3072, 128, 64, 10]
    densities = compute_densities(args.density, sizes)

    # create PyTorch model M1 and Nerva model M2
    optimizer2 = make_nerva_optimizer(args.momentum, args.nesterov)
    M1, M2 = make_models(sizes, densities, optimizer2, args.batch_size)
    optimizer1 = optim.SGD(M1.parameters(), lr=args.learning_rate, momentum=args.momentum, nesterov=args.nesterov)
    loss1 = nn.CrossEntropyLoss()
    loss2 = nerva.loss.SoftmaxCrossEntropyLoss()
    copy_weights_and_biases(M1, M2)

    print(M1)
    print(loss1)
    print(M2)
    print(loss2)

    if args.run != 'nerva':
        train_pytorch(M1, train_loader, test_loader, optimizer1, loss1, args.epochs, args.show)
        print(f'Accuracy of the network on the 10000 test images: {100 * compute_accuracy1(M1, test_loader):.3f} %')

    if args.run != 'pytorch':
        train_nerva(M2, train_loader, test_loader, loss2, args.epochs, args.batch_size, args.learning_rate, args.show)
        print(f'Accuracy of the network on the 10000 test images: {100 * compute_accuracy2(M2, test_loader):.3f} %')


if __name__ == '__main__':
    main()
