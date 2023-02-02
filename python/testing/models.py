import math
import pathlib
import tempfile
from typing import List, Union

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

import nerva.layers
from testing.numpy_utils import load_numpy_arrays_from_npy_file, pp, save_eigen_array, load_eigen_array, l1_norm


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
                save_eigen_array(f, layer.weight.detach().numpy())

    def import_weights(self, filename: str):
        print(f'Importing weights from {filename}')
        with open(filename, "rb") as f:
            for layer in self.layers:
                layer.weight.data = torch.Tensor(load_eigen_array(f))

    def export_bias(self, filename: str):
        with open(filename, "wb") as f:
            for layer in self.layers:
                save_eigen_array(f, layer.bias.detach().numpy())

    def import_bias(self, filename: str):
        print(f'Importing bias from {filename}')
        with open(filename, "rb") as f:
            for layer in self.layers:
                layer.bias.data = torch.Tensor(load_eigen_array(f))

    def print_weight_info(self):
        for i, layer in enumerate(self.layers):
            print(f'|w{i}| = {l1_norm(layer.weight.detach().numpy())}')

    def scale_weights(self, factor):
        print(f'Scale weights with factor {factor}')
        for layer in self.layers:
            layer.weight.data *= factor


# For now, we only support Xavier weights
def create_mask(W: torch.Tensor, density: float):
    x = 1.0 / math.sqrt(W.shape[0])
    x /= density  # compensate for the sparsity
    mask = torch.zeros_like(W)

    for i in range(mask.size(0)):
        for j in range(mask.size(1)):
            if np.random.uniform(0.0, 1.0) < density:
                mask[i, j] = np.random.uniform(-x, x)
    return mask


# Alternative version that uses a more direct way of masking
class MLP1a(nn.Module):
    """ Multi-Layer Perceptron """
    def __init__(self, sizes, densities):
        super().__init__()
        self.sizes = sizes
        self.optimizer = None
        self.masks = None
        n = len(sizes) - 1  # the number of layers
        self.layers = nn.ModuleList()
        for i in range(n):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        self.set_masks(densities)

    def set_masks(self, densities):
        print(f'Setting masks with densities {densities}')
        self.masks = []
        for layer, density in zip(self.layers, densities):
            if density == 1.0:
                self.masks.append(None)
            else:
                self.masks.append(create_mask(layer.weight, density))

    def apply_masks(self):
        for layer, mask in zip(self.layers, self.masks):
            if mask is not None:
                layer.weight.data = layer.weight.data * mask

    def optimize(self):
        self.optimizer.step()
        self.apply_masks()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        x = self.layers[-1](x)  # output layer does not have an activation function
        return x

    def weights(self) -> List[torch.Tensor]:
        return [layer.weight.detach().numpy() for layer in self.layers]

    def bias(self) -> List[torch.Tensor]:
        return [layer.bias.detach().numpy() for layer in self.layers]

    def export_weights(self, filename: str):
        with open(filename, "wb") as f:
            for layer in self.layers:
                save_eigen_array(f, layer.weight.detach().numpy())

    def import_weights(self, filename: str):
        print(f'Importing weights from {filename}')
        with open(filename, "rb") as f:
            for layer in self.layers:
                layer.weight.data = torch.Tensor(load_eigen_array(f))

    def export_bias(self, filename: str):
        with open(filename, "wb") as f:
            for layer in self.layers:
                save_eigen_array(f, layer.bias.detach().numpy())

    def import_bias(self, filename: str):
        print(f'Importing bias from {filename}')
        with open(filename, "rb") as f:
            for layer in self.layers:
                layer.bias.data = torch.Tensor(load_eigen_array(f))

    def print_weight_info(self):
        for i, layer in enumerate(self.layers):
            print(f'|w{i}| = {l1_norm(layer.weight.detach().numpy())}')

    def scale_weights(self, factor):
        print(f'Scale weights with factor {factor}')
        for layer in self.layers:
            layer.weight.data *= factor


class MLP2(nerva.layers.Sequential):
    def __init__(self, sizes, densities, optimizer, batch_size):
        super().__init__()
        self.sizes = sizes
        self.optimizer = optimizer
        self.loss = None
        self.learning_rate = None
        n = len(densities)  # the number of layers
        activations = [nerva.layers.ReLU()] * (n-1) + [nerva.layers.NoActivation()]
        output_sizes = sizes[1:]
        for (density, size, activation) in zip(densities, output_sizes, activations):
            if density == 1.0:
                 self.add(nerva.layers.Dense(size, activation=activation, optimizer=optimizer))
            else:
                self.add(nerva.layers.Sparse(size, density, activation=activation, optimizer=optimizer))
        self.compile(sizes[0], batch_size)

    def weights(self) -> List[np.ndarray]:
        filename = tempfile.NamedTemporaryFile().name + '_weights.npy'
        self.export_weights(filename)
        result = load_numpy_arrays_from_npy_file(filename)
        pathlib.Path(filename).unlink(True)
        return result

    def bias(self) -> List[np.ndarray]:
        def flatten(x: np.ndarray):
            if len(x.shape) == 2 and x.shape[1] == 1:
                return x.reshape(x.shape[0])
            else:
                return x

        filename = tempfile.NamedTemporaryFile().name + '_bias.npy'
        self.export_bias(filename)
        bias = load_numpy_arrays_from_npy_file(filename)
        pathlib.Path(filename).unlink(True)
        # N.B. The shape of the bias can be (128,1), in which case we flatten it to (128).
        return [flatten(b) for b in bias]


def copy_weights_and_biases(model1: Union[MLP1, MLP1a, MLP2], model2: Union[MLP1, MLP2]):
    """
    Copies models and weights from model1 to model2
    :param model1:
    :param model2:
    :return:
    """
    name = tempfile.NamedTemporaryFile().name
    filename1 = name + '_weights.npy'
    filename2 = name + '_bias.npy'
    model1.export_weights(filename1)
    model2.import_weights(filename1)
    model1.export_bias(filename2)
    model2.import_bias(filename2)
    pathlib.Path(filename1).unlink(True)
    pathlib.Path(filename2).unlink(True)


def print_model_info(M):
    W = M.weights()
    b = M.bias()
    for i in range(len(W)):
        pp(f'W{i + 1}', W[i])
        pp(f'b{i + 1}', b[i])
