#!/bin/bash

PYTHONPATH=..
layers="ReLU;ReLU;Linear"
sizes="3072,1024,512,10"
optimizers="Momentum(mu=0.9);Momentum(mu=0.9);Momentum(mu=0.9)"
init_weights="Xavier,Xavier,Xavier"
batch_size=100
epochs=1
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

# create directory logs if it does not exist
mkdir -p logs

train "numpy-colwise"      --numpy      --colwise
train "numpy-rowwise"      --numpy      --rowwise
train "tensorflow-colwise" --tensorflow --colwise
train "tensorflow-rowwise" --tensorflow --rowwise
train "torch-colwise"      --torch      --colwise
train "torch-rowwise"      --torch      --rowwise
train "jax-colwise"        --jax      --colwise
train "jax-rowwise"        --jax      --rowwise

train "numpy-colwise-debug"      --numpy      --colwise --debug
train "numpy-rowwise-debug"      --numpy      --rowwise --debug
train "tensorflow-colwise-debug" --tensorflow --colwise --debug
train "tensorflow-rowwise-debug" --tensorflow --rowwise --debug
train "torch-colwise-debug"      --torch      --colwise --debug
train "torch-rowwise-debug"      --torch      --rowwise --debug
train "jax-colwise-debug"        --jax        --colwise --debug
train "jax-rowwise-debug"        --jax        --rowwise --debug

