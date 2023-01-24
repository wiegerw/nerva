# This code is originally from https://github.com/VITA-Group/Random_Pruning
# There is no license information available

import math
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet_300_100(nn.Module):
    """Simple NN with hidden layers [300, 100]

    Based on https://github.com/mi-lad/snip/blob/master/train.py
    by Milad Alizadeh.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 300, bias=True)
        self.fc2 = nn.Linear(300, 100, bias=True)
        self.fc3 = nn.Linear(100, 10, bias=True)

    def forward(self, x):
        x0 = x.view(-1, 28 * 28)
        x1 = F.relu(self.fc1(x0))
        x2 = F.relu(self.fc2(x1))
        x3 = self.fc3(x2)
        return x3


class MLP_CIFAR10(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(3 * 32 * 32, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x0 = F.relu(self.fc1(x.view(-1, 3 * 32 * 32)))
        x1 = F.relu(self.fc2(x0))
        return F.log_softmax(self.fc3(x1), dim=1)

    def export_weights(self, filename: str):
        with open(filename, "wb") as f:
            W1 = self.fc1.weight.detach().numpy()
            W2 = self.fc2.weight.detach().numpy()
            W3 = self.fc3.weight.detach().numpy()
            np.save(f, np.asfortranarray(W1))
            np.save(f, np.asfortranarray(W2))
            np.save(f, np.asfortranarray(W3))

    def export_bias(self, filename: str):
        with open(filename, "wb") as f:
            b1 = self.fc1.bias.detach().numpy()
            b2 = self.fc2.bias.detach().numpy()
            b3 = self.fc3.bias.detach().numpy()
            np.save(f, np.asfortranarray(b1))
            np.save(f, np.asfortranarray(b2))
            np.save(f, np.asfortranarray(b3))

    def info(self):
        torch.set_printoptions(threshold=10)
        W1 = self.fc1.weight
        W2 = self.fc2.weight
        W3 = self.fc3.weight
        b1 = self.fc1.bias
        b2 = self.fc2.bias
        b3 = self.fc3.bias
        print(f'W1 = {W1.shape}')
        print(W1)
        print(f'b1 = {b1.shape}')
        print(b1)
        print(f'W2 = {W2.shape}')
        print(W2)
        print(f'b2 = {b2.shape}')
        print(b2)
        print(f'W3 = {W3.shape}')
        print(W3)
        print(f'b3 = {b3.shape}')
        print(b3)
