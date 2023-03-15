#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import torch
from torch import nn as nn
from torch.nn import functional as F

import nerva.layers
import nerva.optimizers
from testing.datasets import save_dict_to_npz, load_dict_from_npz
from testing.masking import create_mask


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

    def save_weights(self, filename: str):
        print(f'Saving weights to {filename}')
        data = {}
        for i, layer in enumerate(self.layers):
            data[f'W{i + 1}'] = layer.weight.data
            data[f'b{i + 1}'] = layer.bias.data
        save_dict_to_npz(filename, data)

    def load_weights(self, filename: str):
        data = load_dict_from_npz(filename)
        for i, layer in enumerate(self.layers):
            layer.weight.data = data[f'W{i + 1}']
            layer.bias.data = data[f'b{i + 1}']

    def __str__(self):
        def density_info(layer, mask: torch.Tensor):
            if mask is not None:
                n, N = torch.count_nonzero(mask), mask.numel()
            else:
                n, N = layer.weight.numel(), layer.weight.numel()
            return f'{n}/{N} ({100 * n / N:.8f}%)'

        density_info = [density_info(layer, mask) for layer, mask in zip(self.layers, self.masks)]
        return f'{super().__str__()}\nscheduler = {self.learning_rate}\nlayer densities: {", ".join(density_info)}\n'


class MLPPyTorchTrimmedRelu(MLPPyTorch):
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


class MLPNerva(nerva.layers.Sequential):
    """ Nerva Multilayer perceptron
    """
    def __init__(self, layer_sizes, layer_densities, optimizers, activations, loss, learning_rate, batch_size):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.layer_densities = layer_densities
        self.loss = loss
        self.learning_rate = learning_rate

        n_layers = len(layer_densities)
        assert len(layer_sizes) == n_layers + 1
        assert len(activations) == n_layers
        assert len(optimizers) == n_layers

        output_sizes = layer_sizes[1:]
        for (density, size, activation, optimizer) in zip(layer_densities, output_sizes, activations, optimizers):
            if density == 1.0:
                 self.add(nerva.layers.Dense(size, activation=activation, optimizer=optimizer))
            else:
                self.add(nerva.layers.Sparse(size, density, activation=activation, optimizer=optimizer))

        self.compile(layer_sizes[0], batch_size)

    def save_weights(self, filename: str):
        """
        Saves the weights and biases to a file in .npz format

        The weight matrices should have keys W1, W2, ... and the bias vectors should have keys "b1, b2, ..."
        :param filename: the name of the file
        """
        self.compiled_model.save_weights(filename)

    def load_weights(self, filename: str):
        """
        Loads the weights and biases from a file in .npz format

        The weight matrices are stored using the keys W1, W2, ... and the bias vectors using the keys "b1, b2, ..."
        :param filename: the name of the file
        """
        self.compiled_model.load_weights(filename)

    def __str__(self):
        density_info = [layer.density_info() for layer in self.layers]
        return f'{super().__str__()}\nloss = {self.loss}\nscheduler = {self.learning_rate}\nlayer densities: {", ".join(density_info)}\n'
