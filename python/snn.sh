#!/bin/bash

function train()
{
  logfile="snn-$1.log"
  shift
  extra_args=$*

  python3 snn.py --sizes="3072,1024,512,10" --batch-size=100 --epochs=100 --lr=0.1 --precision=5 --seed=159 --copy $extra_args >& $logfile
}

train "dense-pytorch"            "--torch"
train "dense-nerva"              "--nerva"
train "sparse-pytorch"           "--torch --density=0.05"
train "sparse-nerva"             "--nerva --density=0.05"
train "dense-pytorch-augmented"  "--torch --augmented"
train "dense-nerva-augmented"    "--nerva --augmented"
train "sparse-pytorch-augmented" "--torch --augmented --density=0.05"
train "sparse-nerva-augmented"   "--nerva --augmented --density=0.05"


