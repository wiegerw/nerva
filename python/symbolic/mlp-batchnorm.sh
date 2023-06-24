#!/bin/bash

PYTHONPATH=..
layers="ReLU;BatchNormalization;ReLU;Linear"
sizes="3072,1024,512,10"
optimizers="Momentum(0.9);Momentum(0.9);Momentum(0.9);Momentum(0.9)"
init_weights="Xavier,Xavier,Xavier"
batch_size=100
epochs=2
loss=SoftmaxCrossEntropy
learning_rate="Constant(0.01)"
weights="../mlp-compare.npz"
dataset="../cifar1/epoch0.npz"

function train()
{
  local logfile="logs/$1.log"
  shift
  extra_args=$*
  python3 -u mlp.py \
          --layers=$layers \
          --sizes=$sizes \
          --optimizers=$optimizers \
          --init-weights=$init_weights \
          --batch-size=$batch_size \
          --epochs=$epochs \
          --loss=$loss \
          --learning-rate=$learning_rate \
          --weights=$weights \
          --dataset=$dataset \
          $extra_args \
        	2>&1 | tee $logfile
}

train "numpy-batchnorm-colwise"      --numpy      --colwise
train "numpy-batchnorm-rowwise"      --numpy      --rowwise
train "tensorflow-batchnorm-colwise" --tensorflow --colwise
train "tensorflow-batchnorm-rowwise" --tensorflow --rowwise
train "torch-batchnorm-colwise"      --torch      --colwise
train "torch-batchnorm-rowwise"      --torch      --rowwise
train "jax-batchnorm-colwise"        --jax        --colwise
train "jax-batchnorm-rowwise"        --jax        --rowwise

