#!/bin/bash
source utilities.sh
source mlp-functions.sh

seed=1
init_weights=XavierNormalized
density=0.05
sizes="3072,1024,512,10"
layers="ReLU;ReLU;Linear"
optimizers="Momentum(0.9)"
learning_rate="Constant(0.1)"
loss=SoftmaxCrossEntropy
batch_size=100
epochs=5
name=sparse

tool="../tools/dist/mlp_rowwise"

computation=eigen
#train_cpp --dataset=cifar10

computation=mkl
train_cpp --dataset=cifar10

tool="../tools/dist/mlp_colwise"

computation=eigen
train_cpp --dataset=cifar10

computation=mkl
train_cpp --dataset=cifar10

tool=mlprowwise.py
#train_python --datadir=./data --manual

tool=mlpcolwise.py
train_python --datadir=./data --manual
