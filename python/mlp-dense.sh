#!/bin/bash
source utilities.sh
source mlp-functions.sh

seed=1
init_weights=XavierNormalized
density=1
sizes="3072,1024,512,10"
layers="ReLU;ReLU;Linear"
optimizers="Momentum(0.9)"
learning_rate="Constant(0.1)"
loss=SoftmaxCrossEntropy
batch_size=100
epochs=5
name=default

tool="../tools/dist/mlp_rowwise"
train_cpp --dataset=cifar10

tool="../tools/dist/mlp_rowwise"
train_cpp --dataset=cifar10 --computation=mkl

tool="../tools/dist/mlp_colwise"
train_cpp --dataset=cifar10

tool="../tools/dist/mlp_colwise"
train_cpp --dataset=cifar10 --computation=mkl

tool=mlprowwise.py
train_python --datadir=./data

tool=mlprowwise.py
train_python --datadir=./data --computation=mkl

tool=mlpcolwise.py
train_python --datadir=./data

tool=mlpcolwise.py
train_python --datadir=./data --computation=mkl

name="dense-manual"
tool=mlprowwise.py
train_python --datadir=./data --manual

tool=mlpcolwise.py
train_python --datadir=./data --manual
