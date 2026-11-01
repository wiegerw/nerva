#!/usr/bin/env python

# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

"""Download the MNIST or CIFAR-10 dataset, and save a flattened and normalized version as a .npz file.

The .npz file contains the arrays Xtrain, Ttrain, Xtest and Ttest, where Xtrain and Xtest have
dtype float32 with values in [0, 1], and Ttrain and Ttest contain the class labels.
"""

import argparse
import hashlib
import os
import pickle
import sys
import tarfile
import urllib.request

import numpy as np

MNIST_URL = 'https://s3.amazonaws.com/img-datasets/mnist.npz'
MNIST_SHA256 = '731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1'

CIFAR10_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR10_MD5 = 'c58f30108f718f92721af3b95e74349a'  # as published on https://www.cs.toronto.edu/~kriz/cifar.html

TIMEOUT = 60  # seconds


def file_hash(filepath: str, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, filepath: str, algorithm: str, expected_hash: str) -> bool:
    """Download url to filepath and verify its hash. Returns True on success.

    The file is downloaded to a temporary name first, so that an interrupted
    download does not leave a partial file behind under the final name.
    """
    tmp_filepath = filepath + '.part'
    print(f"Downloading '{url}' to '{filepath}'...")
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as response, open(tmp_filepath, 'wb') as f:
            for chunk in iter(lambda: response.read(1 << 16), b''):
                f.write(chunk)
    except OSError as e:
        print(f"Error downloading '{url}': {e}")
        if os.path.exists(tmp_filepath):
            os.remove(tmp_filepath)
        return False

    actual_hash = file_hash(tmp_filepath, algorithm)
    if actual_hash != expected_hash:
        print(f"Error: the {algorithm} hash of the downloaded file is {actual_hash}, expected {expected_hash}.")
        os.remove(tmp_filepath)
        return False

    os.replace(tmp_filepath, filepath)
    print('Download completed.')
    return True


def download_mnist(data_dir: str = '.') -> bool:
    """Downloads the MNIST dataset to data_dir/mnist.npz if it does not exist yet."""
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, 'mnist.npz')
    if os.path.exists(filepath):
        print(f"MNIST dataset already exists at '{filepath}'. Skipping download.")
        return True
    return download(MNIST_URL, filepath, 'sha256', MNIST_SHA256)


def create_flattened_mnist(data_dir: str = '.', output_filename: str = 'mnist-flattened.npz') -> bool:
    """Loads data_dir/mnist.npz, flattens and normalizes the images, and saves the result to a new .npz file."""
    mnist_filepath = os.path.join(data_dir, 'mnist.npz')
    output_path = os.path.join(data_dir, output_filename)

    if not os.path.exists(mnist_filepath):
        print(f"Error: MNIST dataset file '{mnist_filepath}' not found. Use --download to download it.")
        return False

    if os.path.exists(output_path):
        print(f"Flattened MNIST dataset already exists at '{output_path}'. Skipping flattening.")
        return True

    with np.load(mnist_filepath) as f:
        x_train, Ttrain = f['x_train'], f['y_train']
        x_test, Ttest = f['x_test'], f['y_test']

    # Flatten the images and normalize the pixel values to the range [0, 1]
    Xtrain = x_train.reshape(x_train.shape[0], -1).astype(np.float32) / 255
    Xtest = x_test.reshape(x_test.shape[0], -1).astype(np.float32) / 255

    np.savez_compressed(output_path, Xtrain=Xtrain, Xtest=Xtest, Ttrain=Ttrain, Ttest=Ttest)
    print(f"Flattened and normalized MNIST dataset saved to '{output_path}'")
    return True


def download_cifar10(data_dir: str = '.') -> bool:
    """Downloads and extracts the CIFAR-10 dataset in data_dir if it does not exist yet."""
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, 'cifar-10-python.tar.gz')
    extracted_path = os.path.join(data_dir, 'cifar-10-batches-py')

    if os.path.exists(filepath):
        print(f"File '{filepath}' already exists. Skipping download.")
    elif not download(CIFAR10_URL, filepath, 'md5', CIFAR10_MD5):
        return False

    if os.path.exists(extracted_path):
        print(f"Directory '{extracted_path}' already exists. Skipping extraction.")
        return True

    print('Extracting CIFAR-10 dataset...')
    try:
        with tarfile.open(filepath, 'r:gz') as tar:
            if hasattr(tarfile, 'data_filter'):
                # Reject unsafe archive members, such as absolute paths or links outside data_dir
                tar.extractall(path=data_dir, filter='data')
            else:
                tar.extractall(path=data_dir)
    except (tarfile.TarError, OSError) as e:
        print(f"Error extracting CIFAR-10 dataset: {e}")
        if os.path.exists(extracted_path):
            import shutil
            shutil.rmtree(extracted_path)
        return False
    print('Extraction completed.')
    return True


def load_cifar10_batch(filename: str):
    """Loads a single batch of CIFAR-10 data."""
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    return batch[b'data'], batch[b'labels']


def create_flattened_cifar10(data_dir: str = '.', output_filename: str = 'cifar10-flattened.npz') -> bool:
    """Loads the extracted CIFAR-10 dataset, flattens and normalizes the images, and saves the result to a new .npz file."""
    extracted_dir = os.path.join(data_dir, 'cifar-10-batches-py')
    if not os.path.exists(extracted_dir):
        print(f"Error: CIFAR-10 data directory '{extracted_dir}' not found. Use --download to download it.")
        return False

    output_path = os.path.join(data_dir, output_filename)
    if os.path.exists(output_path):
        print(f"Flattened CIFAR-10 dataset already exists at '{output_path}'. Skipping flattening.")
        return True

    train_data = []
    train_labels = []
    for i in range(1, 6):
        data, labels = load_cifar10_batch(os.path.join(extracted_dir, f'data_batch_{i}'))
        train_data.append(data)
        train_labels.extend(labels)
    test_data, test_labels = load_cifar10_batch(os.path.join(extracted_dir, 'test_batch'))

    Xtrain = np.vstack(train_data)
    Xtest = np.array(test_data)
    Ttrain = np.array(train_labels, dtype=np.int32)
    Ttest = np.array(test_labels, dtype=np.int32)

    # The images are stored as 3x32x32 (channels first); store them as 32x32x3 (channels last), flattened
    Xtrain = Xtrain.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).reshape(Xtrain.shape[0], -1)
    Xtest = Xtest.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).reshape(Xtest.shape[0], -1)

    # Normalize the pixel values to the range [0, 1]
    Xtrain = Xtrain.astype(np.float32) / 255
    Xtest = Xtest.astype(np.float32) / 255

    np.savez_compressed(output_path, Xtrain=Xtrain, Xtest=Xtest, Ttrain=Ttrain, Ttest=Ttest)
    print(f"Flattened and normalized CIFAR-10 dataset saved to '{output_path}'")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and/or flatten the MNIST and CIFAR-10 datasets.")
    parser.add_argument("--dataset", type=str, choices=['mnist', 'cifar10'], help="The dataset to process (mnist or cifar10)")
    parser.add_argument("--download", action="store_true", help="Download the dataset if it does not exist yet")
    parser.add_argument("--data-dir", type=str, default=".", help="Directory to store the datasets (default: .)")
    args = parser.parse_args()

    if not args.dataset:
        parser.print_help()
        return 1

    if args.dataset == 'mnist':
        ok = (not args.download or download_mnist(args.data_dir)) and create_flattened_mnist(args.data_dir)
    else:
        ok = (not args.download or download_cifar10(args.data_dir)) and create_flattened_cifar10(args.data_dir)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
