#!/bin/bash

PYTHONPATH=..
layers="SReLU(al=0, tl=0, ar=0, tr=1);SReLU;Linear"
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
          --layers="$layers" \
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

train "numpy-srelu-colwise"      --numpy      --colwise
train "numpy-srelu-rowwise"      --numpy      --rowwise
train "tensorflow-srelu-colwise" --tensorflow --colwise
train "tensorflow-srelu-rowwise" --tensorflow --rowwise
train "torch-srelu-colwise"      --torch      --colwise
train "torch-srelu-rowwise"      --torch      --rowwise
train "jax-srelu-colwise"        --jax        --colwise
train "jax-srelu-rowwise"        --jax        --rowwise

train "numpy-srelu-colwise-debug"      --numpy      --colwise --debug
train "numpy-srelu-rowwise-debug"      --numpy      --rowwise --debug
train "tensorflow-srelu-colwise-debug" --tensorflow --colwise --debug
train "tensorflow-srelu-rowwise-debug" --tensorflow --rowwise --debug
train "torch-srelu-colwise-debug"      --torch      --colwise --debug
train "torch-srelu-rowwise-debug"      --torch      --rowwise --debug
train "jax-srelu-colwise-debug"        --jax        --colwise --debug
train "jax-srelu-rowwise-debug"        --jax        --rowwise --debug

