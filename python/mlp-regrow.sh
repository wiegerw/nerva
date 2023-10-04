#!/bin/bash
source utilities.sh
source mlp-functions.sh

seed=1
init_weights=XavierNormalized
density=0.05
sizes="3072,1024,512,10"
layers="ReLU;ReLU;Linear"
optimizers="Momentum(0.9)"
learning_rate="Constant(0.01)"
loss=SoftmaxCrossEntropy
batch_size=100
epochs=5
prune="Magnitude(0.2)"
grow=Random
grow_weights=XavierNormalized
name=regrow

tool="../tools/dist/mlp_rowwise"
computation=eigen
#train_cpp --dataset=cifar10 --prune=$prune --grow=$grow --grow-weights=$grow_weights --clip=1e-20
computation=mkl
train_cpp --dataset=cifar10 --prune=$prune --grow=$grow --grow-weights=$grow_weights --clip=1e-20

tool="../tools/dist/mlp_colwise"
computation=eigen
train_cpp --dataset=cifar10 --prune=$prune --grow=$grow --grow-weights=$grow_weights --clip=1e-20
computation=mkl
train_cpp --dataset=cifar10 --prune=$prune --grow=$grow --grow-weights=$grow_weights --clip=1e-20

computation=mkl
tool=mlprowwise.py
train_python --datadir=./data --prune=$prune --grow=$grow --grow-weights=$grow_weights --clip=1e-20
tool=mlpcolwise.py
train_python --datadir=./data --prune=$prune --grow=$grow --grow-weights=$grow_weights --clip=1e-20
