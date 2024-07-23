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
train_cpp --dataset=cifar10 --clip=1e-20

tool="../tools/dist/mlp_rowwise"
train_cpp --dataset=cifar10 --computation=mkl --clip=1e-20

tool="../tools/dist/mlp_colwise"
train_cpp --dataset=cifar10 --clip=1e-20

tool="../tools/dist/mlp_colwise"
train_cpp --dataset=cifar10 --computation=mkl --clip=1e-20

tool=mlp.py
train_python --datadir=./data
train_python --datadir=./data --computation=mkl
name="dense-manual"
train_python --datadir=./data --manual
