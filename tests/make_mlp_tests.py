#!/usr/bin/env python3

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from test_utilities import random_float_matrix, make_target_colwise, make_target_rowwise, print_cpp_matrix_declaration

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class MLP(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        n = len(sizes) - 1
        self.layers = nn.ModuleList()
        for i in range(n):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        x = self.layers[-1](x)
        return x


def train(M, X, T, optimizer, loss_fn, epochs, rowwise):
    N, _ = X.shape  # N is the number of training examples

    for epoch in range(epochs):
        optimizer.zero_grad()
        Y = M(X)
        Y.retain_grad()
        loss = loss_fn(Y, T)
        loss.backward()
        DY = Y.grad.detach()
        print_cpp_matrix_declaration(f'Y{epoch+1}', Y.detach().numpy() if rowwise else Y.detach().numpy().T)
        print_cpp_matrix_declaration(f'DY{epoch+1}', DY.detach().numpy() if rowwise else DY.detach().numpy().T)
        optimizer.step()


def make_testcase(name: str, sizes: list[int], N: int, rowwise=False):
    """
    Makes a test case for C++.
    :param name: The name of the test case.
    :param sizes: The input and output sizes of the layers.
    :param N: The number of samples in the data set.
    """

    print(f'TEST_CASE("{name}")')
    print('{')

    D = sizes[0]   # the input size
    K = sizes[-1]  # the output size
    lr = 0.01      # the learning rate

    X = random_float_matrix(N, D, 0.0, 1.0).astype(np.float32)
    T = make_target_rowwise(N, K).astype(np.float32)  # a one hot encoded target

    print_cpp_matrix_declaration('X', X, rowwise=rowwise)
    print_cpp_matrix_declaration('T', T, rowwise=rowwise)

    X = torch.from_numpy(X)
    T = torch.from_numpy(T)

    device = torch.device("cpu")
    M = MLP(sizes).to(device)

    for i, layer in enumerate(M.layers):
        print_cpp_matrix_declaration(f'W{i+1}', layer.weight.detach().numpy(), rowwise=rowwise)
        print_cpp_matrix_declaration(f'b{i+1}', layer.bias.detach().numpy(), rowwise=rowwise)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(M.parameters(), lr=lr)
    epochs = 2

    train(M, X, T, optimizer, loss_fn, epochs, rowwise)

    print(f'  scalar lr = {lr};')
    print('  test_mlp_execution(X, T, W1, b1, W2, b2, W3, b3, Y1, DY1, Y2, DY2, lr);')
    print('}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert and publish analysis files')
    parser.add_argument('--colwise', help='Generate tests for colwise layout', action='store_true')
    args = parser.parse_args()

    np.random.seed(42)
    np.set_printoptions(precision=6)
    rowwise = not args.colwise

    make_testcase("test_mlp1", sizes=[2, 6, 4, 3], N=5, rowwise=rowwise)
    make_testcase("test_mlp2", sizes=[3, 5, 2, 4], N=4, rowwise=rowwise)
    make_testcase("test_mlp3", sizes=[6, 2, 2, 3], N=8, rowwise=rowwise)
