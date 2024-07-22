#!/usr/bin/env python3

import argparse
import os
from io import StringIO

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from test_utilities import random_float_matrix, make_target_colwise, make_target_rowwise, print_cpp_matrix_declaration, \
    insert_text_in_file

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


def train(out, M, X, T, optimizer, loss_fn, epochs, rowwise):
    N, _ = X.shape  # N is the number of training examples

    for epoch in range(epochs):
        optimizer.zero_grad()
        Y = M(X)
        Y.retain_grad()
        loss = loss_fn(Y, T)
        loss.backward()
        DY = Y.grad.detach()
        print_cpp_matrix_declaration(out, f'Y{epoch+1}', Y.detach().numpy() if rowwise else Y.detach().numpy().T)
        print_cpp_matrix_declaration(out, f'DY{epoch+1}', DY.detach().numpy() if rowwise else DY.detach().numpy().T)
        optimizer.step()


def make_testcase(out: StringIO, name: str, sizes: list[int], N: int, rowwise=False):
    """
    Makes a test case for C++.
    :param name: The name of the test case.
    :param sizes: The input and output sizes of the layers.
    :param N: The number of samples in the data set.
    """

    out.write(f'TEST_CASE("{name}")\n')
    out.write('{\n')

    D = sizes[0]   # the input size
    K = sizes[-1]  # the output size
    lr = 0.01      # the learning rate

    X = random_float_matrix(N, D, 0.0, 1.0).astype(np.float32)
    T = make_target_rowwise(N, K).astype(np.float32)  # a one hot encoded target

    print_cpp_matrix_declaration(out, 'X', X, rowwise=rowwise)
    print_cpp_matrix_declaration(out, 'T', T, rowwise=rowwise)

    X = torch.from_numpy(X)
    T = torch.from_numpy(T)

    device = torch.device("cpu")
    M = MLP(sizes).to(device)

    for i, layer in enumerate(M.layers):
        print_cpp_matrix_declaration(out, f'W{i+1}', layer.weight.detach().numpy(), rowwise=rowwise)
        print_cpp_matrix_declaration(out, f'b{i+1}', layer.bias.detach().numpy(), rowwise=rowwise)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(M.parameters(), lr=lr)
    epochs = 2

    train(out, M, X, T, optimizer, loss_fn, epochs, rowwise)

    out.write(f'  scalar lr = {lr};\n')
    out.write('  test_mlp_execution(X, T, W1, b1, W2, b2, W3, b3, Y1, DY1, Y2, DY2, lr);\n')
    out.write('}\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert and publish analysis files')
    parser.add_argument('--colwise', help='Generate tests for colwise layout', action='store_true')
    args = parser.parse_args()

    np.random.seed(42)
    np.set_printoptions(precision=6)
    rowwise = not args.colwise

    layer_sizes = [[2, 6, 4, 3], [3, 5, 2, 4], [6, 2, 2, 3]]
    example_counts = [5, 4, 8]

    out = StringIO()
    for i in range(3):
        make_testcase(out, f"test_mlp{i+1}", sizes=layer_sizes[i], N=example_counts[i], rowwise=rowwise)
    text = out.getvalue()
    insert_text_in_file('multilayer_perceptron_test.cpp', text)
