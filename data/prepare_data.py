#!/usr/bin/env python

# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import argparse
import os
import requests
import tarfile
import pickle

import numpy as np


def download_mnist(data_dir='data'):
    """
    Downloads the MNIST dataset from the specified URL if it doesn't exist
    in the given directory.

    Args:
        data_dir (str): The directory to save the MNIST dataset.
                        Defaults to 'data'.
    """
    os.makedirs(data_dir, exist_ok=True)
    mnist_filepath = os.path.join(data_dir, 'mnist.npz')
    download_url = 'https://s3.amazonaws.com/img-datasets/mnist.npz'

    if not os.path.exists(mnist_filepath):
        print(f"Downloading MNIST dataset to '{mnist_filepath}'...")
        try:
            response = requests.get(download_url, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes
            with open(mnist_filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("MNIST dataset downloaded successfully.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading MNIST dataset: {e}")
            if os.path.exists(mnist_filepath):
                os.remove(mnist_filepath)
    else:
        print(f"MNIST dataset already exists at '{mnist_filepath}'. Skipping download.")


def create_flattened_mnist(data_dir='data', output_filename='mnist-flattened.npz'):
    """
    Loads the MNIST dataset from 'mnist.npz', flattens and normalizes the
    image data, and saves it to a new .npz file.

    Args:
        data_dir (str): The directory where the MNIST dataset is located.
                        Defaults to 'data'.
        output_filename (str): The name of the output .npz file.
                                 Defaults to 'mnist-flattened.npz'.
    """
    mnist_filepath = os.path.join(data_dir, 'mnist.npz')
    output_path = os.path.join(data_dir, output_filename)

    if not os.path.exists(mnist_filepath):
        print(f"Error: MNIST dataset file '{mnist_filepath}' not found. "
              "Make sure you have downloaded the dataset first.")
        return

    if os.path.exists(output_path):
        print(f"Flattened MNIST dataset already exists at '{output_path}'. Skipping flattening.")
        return

    try:
        # Load the MNIST dataset
        with np.load(mnist_filepath) as f:
            x_train, y_train = f['x_train'], f['y_train']
            x_test, y_test = f['x_test'], f['y_test']

        # Flatten the image data
        num_train_samples = x_train.shape[0]
        num_test_samples = x_test.shape[0]
        img_height = x_train.shape[1]
        img_width = x_train.shape[2]

        Xtrain = x_train.reshape(num_train_samples, img_height * img_width).astype('float32')
        Xtest = x_test.reshape(num_test_samples, img_height * img_width).astype('float32')

        # Normalize the pixel values to the range [0, 1]
        Xtrain /= 255
        Xtest /= 255

        # Assign labels
        Ttrain = y_train
        Ttest = y_test

        # Save the processed data to a new .npz file
        np.savez_compressed(output_path, Xtrain=Xtrain, Xtest=Xtest, Ttrain=Ttrain, Ttest=Ttest)

        print(f"Flattened and normalized MNIST dataset saved to '{output_path}'")

    except Exception as e:
        print(f"An error occurred during MNIST flattening: {e}")


def download_cifar10(data_dir='data'):
    """
    Downloads and extracts the CIFAR-10 dataset if it doesn't already exist.

    Args:
        data_dir (str): The directory to save the CIFAR-10 dataset.
                        Defaults to 'data'.
    """
    os.makedirs(data_dir, exist_ok=True)
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    extracted_dir = "cifar-10-batches-py"

    filepath = os.path.join(data_dir, filename)
    extracted_path = os.path.join(data_dir, extracted_dir)

    # Check if the dataset has already been downloaded
    if os.path.exists(filepath):
        print(f"File '{filepath}' already exists. Skipping download.")
    else:
        print("Downloading CIFAR-10 dataset...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download completed.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading CIFAR-10 dataset: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return

    # Check if the dataset has already been extracted
    if os.path.exists(extracted_path):
        print(f"Directory '{extracted_path}' already exists. Skipping extraction.")
    else:
        print("Extracting CIFAR-10 dataset...")
        try:
            with tarfile.open(filepath, 'r:gz') as tar:
                tar.extractall(path=data_dir)
            print("Extraction completed.")
        except tarfile.TarError as e:
            print(f"Error extracting CIFAR-10 dataset: {e}")
            if os.path.exists(extracted_path):
                import shutil
                shutil.rmtree(extracted_path)
            return

    print("CIFAR-10 setup is complete.")


def load_cifar10_batch(filename):
    """Loads a single batch of CIFAR-10 data."""
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    data = batch[b'data']
    labels = batch[b'labels']
    return data, labels


def create_flattened_cifar10(data_dir='data', output_filename='cifar10-flattened.npz'):
    """
    Loads the downloaded CIFAR-10 dataset, flattens and normalizes the
    image data, and saves it to a new .npz file.

    Args:
        data_dir (str): The directory where the CIFAR-10 dataset is located.
                        Defaults to 'data'.
        output_filename (str): The name of the output .npz file.
                                 Defaults to 'cifar10-flattened.npz'.
    """
    extracted_dir = os.path.join(data_dir, 'cifar-10-batches-py')
    if not os.path.exists(extracted_dir):
        print(f"Error: CIFAR-10 data directory '{extracted_dir}' not found. "
              "Make sure you have downloaded and extracted the dataset using the "
              "`download_cifar10` function.")
        return

    output_path = os.path.join(data_dir, output_filename)
    if os.path.exists(output_path):
        print(f"Flattened CIFAR-10 dataset already exists at '{output_path}'. Skipping flattening.")
        return

    train_data = []
    train_labels = []
    for i in range(1, 6):
        filename = os.path.join(extracted_dir, f'data_batch_{i}')
        data, labels = load_cifar10_batch(filename)
        train_data.append(data)
        train_labels.extend(labels)

    test_filename = os.path.join(extracted_dir, 'test_batch')
    test_data, test_labels = load_cifar10_batch(test_filename)

    Xtrain = np.vstack(train_data).astype('float32')
    Ttrain = np.array(train_labels).astype('int32')
    Xtest = np.array(test_data).astype('float32')
    Ttest = np.array(test_labels).astype('int32')

    # Reshape and normalize
    num_train_samples = Xtrain.shape[0]
    num_test_samples = Xtest.shape[0]

    Xtrain = Xtrain.reshape(num_train_samples, 3, 32, 32).transpose(0, 2, 3, 1)
    Xtest = Xtest.reshape(num_test_samples, 3, 32, 32).transpose(0, 2, 3, 1)

    Xtrain_flattened = Xtrain.reshape(num_train_samples, -1)
    Xtest_flattened = Xtest.reshape(num_test_samples, -1)

    Xtrain_normalized = Xtrain_flattened / 255.0
    Xtest_normalized = Xtest_flattened / 255.0

    # Save the processed data to a new .npz file
    np.savez_compressed(output_path, Xtrain=Xtrain_normalized, Xtest=Xtest_normalized, Ttrain=Ttrain, Ttest=Ttest)

    print(f"Flattened and normalized CIFAR-10 dataset saved to '{output_path}'")


def main():
    parser = argparse.ArgumentParser(description="Download and/or flatten MNIST and CIFAR10 datasets.")
    parser.add_argument("--dataset", type=str, choices=['mnist', 'cifar10'], help="Specify the dataset to process (mnist or cifar10)")
    parser.add_argument("--download", action="store_true", help="Download the specified dataset if it doesn't exist")
    parser.add_argument("--data-dir", type=str, default=".", help="Directory to store the datasets (default: .)")

    args = parser.parse_args()

    if args.dataset:
        if args.dataset == 'mnist':
            if args.download:
                download_mnist(args.data_dir)
            create_flattened_mnist(args.data_dir)
        elif args.dataset == 'cifar10':
            if args.download:
                download_cifar10(args.data_dir)
            create_flattened_cifar10(args.data_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

