#!/usr/bin/env python3

import numpy as np
import nervalib
from testing.numpy_utils import load_eigen_array, save_eigen_array, pp


def export_array(filename: str, x: np.ndarray):
    with open(filename, "wb") as f:
        x = x.reshape(x.shape[1], x.shape[0], order='F').T
        np.save(f, x, allow_pickle=True)


def import_array(filename: str):
    with open(filename, "rb") as f:
        x = np.load(f, allow_pickle=True)
        return x.reshape(x.shape[1], x.shape[0], order='C').T


def test_import_export():
    nervalib.export_default_matrices_to_numpy('D1.npy', 'D2.npy')
    nervalib.import_matrix_from_numpy('D1.npy')
    nervalib.import_matrix_from_numpy('D2.npy')

    D1 = import_array('D1.npy')
    D2 = import_array('D2.npy')
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert (D1 == A).all()
    assert (D2 == A.T).all()

    export_array('E1.npy', D1)
    export_array('E2.npy', D2)
    F1 = import_array('E1.npy')
    F2 = import_array('E2.npy')
    assert (F1 == A).all()
    assert (F2 == A.T).all()

    nervalib.import_matrix_from_numpy('D1.npy')
    nervalib.import_matrix_from_numpy('D2.npy')
    nervalib.import_matrix_from_numpy('E1.npy')
    nervalib.import_matrix_from_numpy('E2.npy')


def test_load_save():
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    pp('A', A)

    with open('A.npy', "wb") as f:
        save_eigen_array(f, A)

    with open('A.npy', "rb") as f:
        B = load_eigen_array(f)

    pp('B', B)
    assert (A == B).all()


if __name__ == '__main__':
    test_import_export()
    test_load_save()