#!/bin/bash
source utilities.sh
source mlp-functions.sh

seed=1
init_weights=Xavier
density=1
sizes="3072,1024,512,10"
layers="ReLU;ReLU;Linear"
dropouts="0.3,0,0"
optimizers="Momentum(0.9)"
learning_rate="Constant(0.01)"
loss=SoftmaxCrossEntropy
batch_size=100
epochs=5
name=dropout

tool="../tools/dist/mlp_rowwise"
train_cpp --dataset=cifar10

tool="../tools/dist/mlp_rowwise"
train_cpp --dataset=cifar10 --computation=mkl

tool="../tools/dist/mlp_colwise"
train_cpp --dataset=cifar10

tool="../tools/dist/mlp_colwise"
train_cpp --dataset=cifar10 --computation=mkl

tool=mlprowwise.py
train_python --datadir=./data --manual

tool=mlpcolwise.py
train_python --datadir=./data --manual
