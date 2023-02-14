from typing import Union

import numpy as np
import torch


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return np.asfortranarray(x.detach().numpy().T)


def to_one_hot_numpy(x: np.ndarray, n_classes: int):
    return flatten_numpy(np.asfortranarray(np.eye(n_classes)[x].T))


def flatten_numpy(x: np.ndarray) -> np.ndarray:
    shape = x.shape
    return x.reshape(shape[0], -1)


def l1_norm(x: np.ndarray):
    return np.abs(x).sum()


def l2_norm(x: np.ndarray):
    return np.linalg.norm(x)


def normalize_image_data(X: np.array, mean=None, std=None):
    if not mean:
        mean = X.mean(axis=(0, 1, 2))
    if not std:
        std = X.std(axis=(0, 1, 2))
    return (X  - mean) / std


def torch_inf_norm(x: torch.Tensor):
    return torch.abs(x).max().item()


def pp(name: str, x: Union[torch.Tensor, np.ndarray]):
    if isinstance(x, np.ndarray):
        x = torch.Tensor(x)
    if len(x.shape) == 1:
        print(f'{name} ({x.shape[0]}) norm = {torch_inf_norm(x)}\n{x.data}')
    else:
        print(f'{name} ({x.shape[0]}x{x.shape[1]}) norm = {torch_inf_norm(x)}\n{x.data}')
