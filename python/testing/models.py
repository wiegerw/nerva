from typing import List

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

import nerva.layers
from testing.datasets import save_dict_to_npz, load_dict_from_npz
from testing.masking import create_mask
from testing.numpy_utils import pp, l1_norm


# This PyTorch model supports sparse layers using binary masks
class MLP1(nn.Module):
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
        self.masks = []
        for layer, density in zip(self.layers, densities):
            if density == 1.0:
                self.masks.append(None)
            else:
                self.masks.append(create_mask(layer.weight, int(density * layer.weight.numel())))

    def apply_masks(self):
        for layer, mask in zip(self.layers, self.masks):
            if mask is not None:
                layer.weight.data = layer.weight.data * mask

    def print_masks(self):
        print('--- masks ---')
        for i, mask in enumerate(self.masks):
            if mask is not None:
                pp(f'mask{i + 1}', mask.int())


    def optimize(self):
        self.apply_masks()  # N.B. This seems to be the correct order
        self.optimizer.step()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        x = self.layers[-1](x)  # output layer does not have an activation function
        return x

    def weights(self) -> List[torch.Tensor]:
        return [layer.weight.detach().numpy() for layer in self.layers]

    def bias(self) -> List[torch.Tensor]:
        return [layer.bias.detach().numpy() for layer in self.layers]

    def export_weights_npz(self, filename: str):
        print(f'Exporting weights to {filename}')
        data = {}
        for i, layer in enumerate(self.layers):
            data[f'W{i + 1}'] = layer.weight.data
            data[f'b{i + 1}'] = layer.bias.data
        save_dict_to_npz(filename, data)

    def import_weights_npz(self, filename: str):
        data = load_dict_from_npz(filename)
        for i, layer in enumerate(self.layers):
            layer.weight.data = data[f'W{i + 1}']
            layer.bias.data = data[f'b{i + 1}']

    def print_weight_info(self):
        for i, layer in enumerate(self.layers):
            print(f'|w{i + 1}| = {l1_norm(layer.weight.detach().numpy())}')

    def scale_weights(self, factor):
        print(f'Scale weights with factor {factor}')
        for layer in self.layers:
            layer.weight.data *= factor

    def info(self):
        def density_info(layer, mask: torch.Tensor):
            if mask is not None:
                n, N = torch.count_nonzero(mask), mask.numel()
            else:
                n, N = layer.weight.numel(), layer.weight.numel()
            return f'{n}/{N} ({100 * n / N:.3f}%)'

        print('=== PyTorch model ===')
        print(self)
        print(f'scheduler = {self.learning_rate}')
        density_info = [density_info(layer, mask) for layer, mask in zip(self.layers, self.masks)]
        print(f'layer densities: {", ".join(density_info)}\n')


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
        return self.compiled_model.weights()

    def bias(self) -> List[np.ndarray]:
        return [b.reshape((b.shape[0])) for b in self.compiled_model.bias()]

    def export_weights_npz(self, filename: str):
        self.compiled_model.export_weights_npz(filename)

    def import_weights_npz(self, filename: str):
        self.compiled_model.import_weights_npz(filename)

    def info(self):
        print('=== Nerva python model ===')
        print(self)
        print(f'loss = {self.loss}')
        print(f'scheduler = {self.learning_rate}')
        density_info = [layer.density_info() for layer in self.layers]
        print(f'layer densities: {", ".join(density_info)}\n')


def print_model_info(M):
    W = M.weights()
    b = M.bias()
    for i in range(len(W)):
        pp(f'W{i + 1}', W[i])
        pp(f'b{i + 1}', b[i])
