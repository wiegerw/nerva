#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from typing import List, Tuple

import jax.numpy as jnp
import sklearn.datasets as dt

from mlps.nerva_jax.activation_functions import ReLUActivation
from mlps.nerva_jax.datasets import MemoryDataLoader
from mlps.nerva_jax.layers_rowwise import ActivationLayer, LinearLayer
from mlps.nerva_jax.learning_rate import MultiStepLRScheduler
from mlps.nerva_jax.loss_functions_rowwise import SoftmaxCrossEntropyLossFunction
from mlps.nerva_jax.multilayer_perceptron_rowwise import MultilayerPerceptron
from mlps.nerva_jax.training_rowwise import sgd


def generate_synthetic_dataset(num_train_samples, num_test_samples, num_features, num_classes, num_redundant=2, class_sep=0.8, random_state=None):
    X, T = dt.make_classification(
        n_samples=num_train_samples + num_test_samples,
        n_features=num_features,
        n_informative=int(0.7 * num_features),
        n_classes=num_classes,
        n_redundant=num_redundant,
        class_sep=class_sep,
        random_state=random_state
    )

    # Split the dataset into a training and test set
    train_batch = jnp.array(range(0, num_train_samples))
    test_batch = jnp.array(range(num_train_samples, num_train_samples + num_test_samples))
    return X[train_batch], T[train_batch], X[test_batch], T[test_batch]


def create_mlp(sizes: List[Tuple[int, int]]):
    M = MultilayerPerceptron()

    for i, (input_size, output_size) in enumerate(sizes):
        if i == len(sizes) - 1:
            layer = LinearLayer(input_size, output_size)
        else:
            layer = ActivationLayer(input_size, output_size, ReLUActivation())
        layer.set_optimizer('Momentum(0.9)')
        layer.set_weights('Xavier')
        M.layers.append(layer)

    return M


def main():
    num_train_samples = 50000
    num_test_samples = 10000
    num_features = 8
    num_classes = 5
    batch_size = 100

    Xtrain, Ttrain, Xtest, Ttest = generate_synthetic_dataset(num_train_samples, num_test_samples, num_features, num_classes)
    train_loader = MemoryDataLoader(Xtrain, Ttrain, batch_size=batch_size, num_classes=num_classes)
    test_loader = MemoryDataLoader(Xtest, Ttest, batch_size=batch_size, num_classes=num_classes)

    M = create_mlp([(num_features, 200), (200, 200), (200, num_classes)])
    loss = SoftmaxCrossEntropyLossFunction()
    epochs = 20
    learning_rate = MultiStepLRScheduler(lr=0.1, milestones=[10, 15], gamma=0.3)
    sgd(M, epochs, loss, learning_rate, train_loader, test_loader)


if __name__ == '__main__':
    main()
