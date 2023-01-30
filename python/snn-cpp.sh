#!/bin/bash

function train_sparse()
{
  density=$1
  shift
  logfile="snn/nerva-sparse-$density.log"

  ../tools/dist/mlpf --epochs=100 --architecture=RRL --hidden="1024,512" --weights=xxx --dataset=cifar10 --learning-rate="multistep_lr(0.1;50,75;0.1)" --size=50000 --loss="softmax-cross-entropy" --algorithm=minibatch --batch-size=100 --normalize --optimizer="nesterov(0.9)" --threads=4 -v --seed=1885661379 --density=$density >& $logfile
}

# create logfile directory
if [ ! -d snn ]; then
  mkdir snn
fi

# dense experiment
../tools/dist/mlpf --epochs=100 --architecture=RRL --hidden="1024,512" --weights=xxx --dataset=cifar10 --learning-rate="multistep_lr(0.1;50,75;0.1)" --size=50000 --loss="softmax-cross-entropy" --algorithm=minibatch --batch-size=100 --normalize --optimizer="nesterov(0.9)" --threads=4 -v --seed=1885661379 >& snn/nerva-dense.log

# sparse experiments
train_sparse 0.001
train_sparse 0.005
train_sparse 0.01
train_sparse 0.05
train_sparse 0.10
train_sparse 0.50
