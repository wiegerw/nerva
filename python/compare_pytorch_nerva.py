#!/usr/bin/env python3

import argparse
import os
import tempfile
from timeit import default_timer as timer
from typing import Union
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


def flatten_torch(X: torch.Tensor) -> torch.Tensor:
    shape = X.shape
    return X.reshape(shape[0], -1)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return np.asfortranarray(flatten_torch(x).detach().numpy().T)


def to_one_hot(x: torch.Tensor) -> np.ndarray:
    return to_numpy(torch.nn.functional.one_hot(x, num_classes = 10).float())


def create_cifar10_dataloaders(batch_size, test_batch_size, datadir='_dataset'):
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
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
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


def generate_pytorch_batches(Xdata: torch.Tensor, Tdata: torch.Tensor, batch_size: int):
    N = Xdata.shape[0]   # N is the number of examples
    K = N // batch_size  # K is the number of batches
    for k in range(K):
        batch = range(k * batch_size, (k + 1) * batch_size)
        yield Xdata[batch], Tdata[batch]


def generate_numpy_batches(Xdata: torch.Tensor, Tdata: torch.Tensor, batch_size: int):
    N = Xdata.shape[0]   # N is the number of examples
    K = N // batch_size  # K is the number of batches
    for k in range(K):
        batch = range(k * batch_size, (k + 1) * batch_size)
        yield to_numpy(Xdata[batch]), to_numpy(Tdata[batch])


def generate_one_hot_numpy_batches(Xdata: torch.Tensor, Tdata: torch.Tensor, batch_size: int):
    N = Xdata.shape[0]   # N is the number of examples
    K = N // batch_size  # K is the number of batches
    for k in range(K):
        batch = range(k * batch_size, (k + 1) * batch_size)
        yield to_numpy(Xdata[batch]), to_one_hot(Tdata[batch])


def normalize_cifar_data(X: np.array, mean=None, std=None):
    if not mean:
        mean = X.mean(axis=(0, 1, 2))
    if not std:
        std = X.std(axis=(0, 1, 2))
    return (X  - mean) / std


def flatten_cifar_data(X: np.array):
    shape = X.shape
    return X.reshape(shape[0], -1)


def load_cifar10_data(show: bool):
    if not os.path.exists("./data"):
        os.makedirs("./data")

    if not os.path.exists("./models"):
        os.makedirs("./models")

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

    if show:
        print('Xtrain=\n', trainset.data.reshape(trainset.data.shape[0], -1))
    Xtrain = normalize_cifar_data(trainset.data / 255, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    Xtrain = flatten_cifar_data(Xtrain)
    Xtrain = torch.Tensor(Xtrain)
    Ttrain = torch.LongTensor(trainset.targets)
    if show:
        print('Xtrain=\n', Xtrain)

    Xtest = normalize_cifar_data(testset.data / 255)
    Xtest = flatten_cifar_data(Xtest)
    Xtest = torch.Tensor(Xtest)
    Ttest = torch.LongTensor(testset.targets)

    return Xtrain, Ttrain, Xtest, Ttest


def compute_accuracy1(M: nn.Module, Xdata: torch.Tensor, Tdata: torch.Tensor, batch_size):
    N = Xdata.shape[0]  # N is the number of examples
    total_correct = 0
    for X, T in generate_pytorch_batches(Xdata, Tdata, batch_size):
        Y = M(X)
        predicted = Y.argmax(axis=1)  # the predicted classes for the batch
        total_correct += (predicted == T).sum().item()
    return total_correct / N


def compute_loss1(M: nn.Module, loss, Xdata: torch.Tensor, Tdata: torch.Tensor, batch_size):
    N = Xdata.shape[0]   # N is the number of examples
    total_loss = 0.0
    for X, T in generate_pytorch_batches(Xdata, Tdata, batch_size):
        Y = M(X)
        total_loss += loss(Y, T).sum()
    return batch_size * total_loss / N


def compute_accuracy2(M: nerva.layers.Sequential, Xdata: torch.Tensor, Tdata: torch.Tensor, batch_size):
    N = Xdata.shape[0]  # N is the number of examples
    total_correct = 0
    for X, T in generate_numpy_batches(Xdata, Tdata, batch_size):
        Y = M.feedforward(X)
        predicted = Y.argmax(axis=0)  # the predicted classes for the batch
        total_correct += (predicted == T).sum().item()
    return total_correct / N


def compute_loss2(M: nerva.layers.Sequential, loss, Xdata: torch.Tensor, Tdata: torch.Tensor, batch_size):
    N = Xdata.shape[0]  # N is the number of examples
    total_loss = 0.0
    for X, T in generate_one_hot_numpy_batches(Xdata, Tdata, batch_size):
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


def train_pytorch(M, Xtrain, Ttrain, Xtest, Ttest, optimizer, criterion, epochs, batch_size, show: bool):
    print('Training...')
    N = Xtrain.shape[0]
    for epoch in range(epochs):
        start = timer()
        for k, (X, T) in enumerate(generate_pytorch_batches(Xtrain, Ttrain, batch_size)):
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
              f'loss: {compute_loss1(M, criterion, Xtrain, Ttrain, batch_size):.3f}  '
              f'train accuracy: {compute_accuracy1(M, Xtrain, Ttrain, batch_size):.3f}  '
              f'test accuracy: {compute_accuracy1(M, Xtest, Ttest, batch_size):.3f}  '
              f'time: {elapsed:.3f}'
             )


def train_nerva(M, Xtrain, Ttrain, Xtest, Ttest, criterion, epochs, batch_size, lr, show: bool):
    print('Training...')
    N = Xtrain.shape[0]
    for epoch in range(epochs):  # loop over the dataset multiple times
        start = timer()
        for k, (X, T) in enumerate(generate_one_hot_numpy_batches(Xtrain, Ttrain, batch_size)):
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
              f'loss: {compute_loss2(M, criterion, Xtrain, Ttrain, batch_size):.3f}  '
              f'train accuracy: {compute_accuracy2(M, Xtrain, Ttrain, batch_size):.3f}  '
              f'test accuracy: {compute_accuracy2(M, Xtest, Ttest, batch_size):.3f}  '
              f'time: {elapsed:.3f}'
             )


def make_nerva_optimizer(momentum=0.0, nesterov=False) -> nerva.optimizers.Optimizer:
    if nesterov:
        return nerva.optimizers.Nesterov(momentum)
    elif momentum > 0.0:
        return nerva.optimizers.Momentum(momentum)
    else:
        return nerva.optimizers.GradientDescent()


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
    cmdline_parser.add_argument('--momentum', type=float, default=0.9, help='the momentum value (default off)')
    cmdline_parser.add_argument("--nesterov", help="apply nesterov", action="store_true")
    args = cmdline_parser.parse_args()

    if args.seed:
        torch.manual_seed(args.seed)

    torch.set_printoptions(precision=args.precision, edgeitems=args.edgeitems, threshold=5)
    Xtrain, Ttrain, Xtest, Ttest = load_cifar10_data(args.show)
    sizes = [3072, 128, 64, 10]

    # create PyTorch model
    M1 = MLP1(sizes)
    loss1 = nn.CrossEntropyLoss()
    optimizer1 = optim.SGD(M1.parameters(), lr=args.learning_rate, momentum=args.momentum, nesterov=args.nesterov)
    print(M1)
    print(loss1)

    # create Nerva model
    optimizer2 = make_nerva_optimizer(args.momentum, args.nesterov)
    M2 = MLP2(sizes, optimizer2, args.batch_size)
    loss2 = nerva.loss.SoftmaxCrossEntropyLoss()
    copy_weights_and_biases(M1, M2)
    print(M2)
    print(loss2)

    if args.run != 'nerva':
        train_pytorch(M1, Xtrain, Ttrain, Xtest, Ttest, optimizer1, loss1, args.epochs, args.batch_size, args.show)
        print(f'Accuracy of the network on the 10000 test images: {100 * compute_accuracy1(M1, Xtest, Ttest, args.batch_size):.3f} %')

    if args.run != 'pytorch':
        train_nerva(M2, Xtrain, Ttrain, Xtest, Ttest, loss2, args.epochs, args.batch_size, args.learning_rate, args.show)
        print(f'Accuracy of the network on the 10000 test images: {100 * compute_accuracy2(M2, Xtest, Ttest, args.batch_size):.3f} %')


if __name__ == '__main__':
    main()
