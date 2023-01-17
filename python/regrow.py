#!/usr/bin/env python3

# This code is from https://github.com/Shiweiliuiiiiiii/SET-MLP-ONE-MILLION-NEURONS/blob/master/set_mlp_sparse_data_structures.py

import tempfile
from typing import List
import numpy as np
from scipy.sparse import csr_matrix
from weights_evolution import weights_evolution_II


# Loads a number of Numpy arrays from an .npy file and returns them in a list
def load_numpy_arrays_from_npy_file(filename: str) -> List[np.ndarray]:
    arrays = []
    try:
        with open(filename, "rb") as f:
            while True:
                arrays.append(np.load(f, allow_pickle=True))
    except IOError:
        pass
    return arrays


# Regrow weights using the function weights_evolution_II
def regrow_weights(M, zeta):
    filename = tempfile.NamedTemporaryFile().name
    M.export_weights(filename)
    weight_matrices = load_numpy_arrays_from_npy_file(filename)
    with open(filename, "wb") as f:
        for W in weight_matrices:
            W = csr_matrix(W)
            W = weights_evolution_II(W, zeta)
            W = W.todense()
            np.save(f, W)
    M.import_weights(filename)
