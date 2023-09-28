#!/bin/bash
source utilities.sh
source mlp-functions.sh

seed=1
init_weights=XavierNormalized
density=0.05
sizes="3072,1024,512,10"
layers="ReLU;ReLU;Linear"
optimizers="Momentum(0.9)"
learning_rate="Constant(lr=0.1)"
loss=SoftmaxCrossEntropy
batch_size=100
epochs=5
name=preprocessed

tool="../tools/dist/mlp_rowwise_eigen"
train_cpp --preprocessed=cifar$seed

tool="../tools/dist/mlp_rowwise_mkl"
train_cpp --preprocessed=cifar$seed

tool="../tools/dist/mlp_colwise_eigen"
train_cpp --preprocessed=cifar$seed

tool="../tools/dist/mlp_colwise_mkl"
train_cpp --preprocessed=cifar$seed

tool=mlprowwise.py
train_python --preprocessed=cifar$seed

tool=mlpcolwise.py
train_python --preprocessed=cifar$seed

name="preprocessed-manual"
tool=mlprowwise.py
train_python --preprocessed=cifar$seed --manual

tool=mlpcolwise.py
train_python --preprocessed=cifar$seed --manual
