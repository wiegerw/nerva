from typing import List, Union

import numpy as np
import torch

# Save it in a format compatible with Nerva C++
def to_eigen(x: np.ndarray):
    if len(x.shape) == 2:
        return x.reshape(x.shape[1], x.shape[0], order='F').T
    return x


def from_eigen(x: np.ndarray):
    if len(x.shape) == 2:
        return x.reshape(x.shape[1], x.shape[0], order='C').T
    return x


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


def save_numpy_arrays_to_npy_file(filename: str, arrays: List[np.ndarray]):
    with open(filename, "wb") as f:
        for array in arrays:
            np.save(f, to_eigen(array), allow_pickle=True)


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


def pp(name: str, x: Union[torch.Tensor, np.ndarray]):
    if isinstance(x, np.ndarray):
        x = torch.Tensor(x.T)
    if len(x.shape) == 1:
        print(f'{name} ({x.shape[0]})\n{x.data}')
    else:
        print(f'{name} ({x.shape[0]}x{x.shape[1]})\n{x.data}')
