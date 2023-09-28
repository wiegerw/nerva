#!/bin/bash

# This script compares the computations of Nerva C++ and Nerva python by
# logging intermediate results.

source utilities.sh
source mlp-functions.sh

seed=1
init_weights=XavierNormalized
density=1.0
sizes="3072,1024,512,10"
layers="ReLU;ReLU;Linear"
optimizers="Momentum(0.9)"
learning_rate="Constant(lr=0.1)"
loss=SoftmaxCrossEntropy
batch_size=100
epochs=1
optimizers="GradientDescent"

function train()
{
  name="compare-$optimizers"
  local weight_file="data/mlp-compare.npz"

  tool="mlptorch.py"
  train_torch --preprocessed=cifar$seed --save-weights=$weight_file --debug

  tool="../tools/dist/mlp_rowwise_eigen"
  train_cpp --preprocessed=cifar$seed --load-weights=$weight_file --debug

  tool="../tools/dist/mlp_rowwise_mkl"
  train_cpp --preprocessed=cifar$seed --load-weights=$weight_file --debug

  tool="../tools/dist/mlp_colwise_eigen"
  train_cpp --preprocessed=cifar$seed --load-weights=$weight_file --debug

  tool="../tools/dist/mlp_colwise_mkl"
  train_cpp --preprocessed=cifar$seed --load-weights=$weight_file --debug

  tool=mlprowwise.py
  train_python --preprocessed=cifar$seed --load-weights=$weight_file --debug --manual

  tool=mlpcolwise.py
  train_python --preprocessed=cifar$seed --load-weights=$weight_file --debug --manual
}

optimizers="GradientDescent"
train

optimizers="Momentum(0.9)"
learning_rate="Constant(0.01)"
train

learning_rate="Constant(0.01)"
optimizers="Nesterov(0.9)"
train
