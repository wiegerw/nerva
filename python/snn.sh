#!/bin/bash

epochs=100
batchsize=100
momentum=0.9

densesizes="1024,512"
densearchitecture=RRL
denseweights=xxx

sparsesizes="1024,1024,1024"
sparsearchitecture=RRRL
sparseweights=xxxx

datadir="./data"

function train_sparse_torch()
{
  local seed=$1
  local augmented=$2
  local lr=$3
  local density=$4

  if [ "$augmented" = "true" ]
  then
       augmentedflag="--augmented"
       logfile="snn/torch-sparse-$density-augmented-seed$seed.log"
  else
       augmentedflag=""
       logfile="snn/torch-sparse-$density-seed$seed.log"
  fi

  python3 snn.py --torch --seed=$seed --density=$density --lr=$lr $augmentedflag --sizes="3072,$sparsesizes,10" \
                 --batch-size=$batchsize --epochs=$epochs --momentum=$momentum --nesterov --datadir="$datadir" \
                 >& $logfile
}

function train_sparse_nerva()
{
  local seed=$1
  local augmented=$2
  local lr=$3
  local density=$4

  if [ "$augmented" = "true" ]
  then
       augmentedflag="--augmented"
       logfile="snn/nerva-sparse-$density-augmented-seed$seed.log"
  else
       augmentedflag=""
       logfile="snn/nerva-sparse-$density-seed$seed.log"
  fi

  ../tools/dist/mlpf --seed=$seed --density=$density $augmentedflag --hidden="$sparsesizes" --batch-size=$batchsize \
                     --epochs=$epochs --learning-rate="multistep_lr($lr;50,75;0.1)" --optimizer="nesterov($momentum)"  \
                     --architecture=$sparsearchitecture --weights=$sparseweights --dataset=cifar10 --size=50000 \
                     --loss="softmax-cross-entropy" --algorithm=minibatch --threads=4 -v --datadir="$datadir" \
                     >& $logfile
}

function train_dense_torch()
{
  local seed=$1
  local augmented=$2
  local lr=$3

  if [ "$augmented" = "true" ]
  then
       augmentedflag="--augmented"
       logfile="snn/torch-dense-augmented-seed$seed.log"
  else
       augmentedflag=""
       logfile="snn/torch-dense-seed$seed.log"
  fi

  python3 snn.py --torch --seed=$seed --lr=$lr $augmentedflag --sizes="3072,$densesizes,10" --batch-size=$batchsize \
                 --epochs=$epochs --momentum=$momentum --nesterov --datadir="$datadir" \
                 >& $logfile
}

function train_dense_nerva()
{
  local seed=$1
  local augmented=$2
  local lr=$3

  if [ "$augmented" = "true" ]
  then
       augmentedflag="--augmented"
       logfile="snn/nerva-dense-augmented-seed$seed.log"
  else
       augmentedflag=""
       logfile="snn/nerva-dense-seed$seed.log"
  fi

  ../tools/dist/mlpf --seed=$seed $augmentedflag --hidden="$densesizes" --batch-size=$batchsize \
                     --epochs=$epochs --learning-rate="multistep_lr($lr;50,75;0.1)" --optimizer="nesterov($momentum)"  \
                     --architecture=$densearchitecture --weights=$denseweights --dataset=cifar10 --size=50000 \
                     --loss="softmax-cross-entropy" --algorithm=minibatch --threads=4 -v --datadir="$datadir" \
                     >& $logfile
}

function train_all()
{
    for seed in 1 2 3 4
    do
        for augmented in true false
        do
            if [ "$augmented" = "true" ]
            then
                datadir="./cifar$seed"
            else
                datadir="./data"
            fi

            train_dense_torch  $seed $augmented 0.01
            train_dense_nerva  $seed $augmented 0.01
            train_sparse_torch $seed $augmented 0.1 0.001
            train_sparse_nerva $seed $augmented 0.1 0.001
            train_sparse_torch $seed $augmented 0.1 0.005
            train_sparse_nerva $seed $augmented 0.1 0.005
            train_sparse_torch $seed $augmented 0.1 0.01
            train_sparse_nerva $seed $augmented 0.1 0.01
            train_sparse_torch $seed $augmented 0.1 0.05
            train_sparse_nerva $seed $augmented 0.1 0.05
            train_sparse_torch $seed $augmented 0.1 0.1
            train_sparse_nerva $seed $augmented 0.1 0.1
            train_sparse_torch $seed $augmented 0.1 0.2
            train_sparse_nerva $seed $augmented 0.1 0.2
            train_sparse_torch $seed $augmented 0.1 0.5
            train_sparse_nerva $seed $augmented 0.1 0.5
        done
    done
}

# used for quick testing
function train_one()
{
    epochs=1
    densesizes="64,32"
    densearchitecture=RRL
    denseweights=xxx

    sparsesizes="64,64"
    sparsearchitecture=RRL
    sparseweights=xxx

    for seed in 1
    do
        for augmented in false
        do
            train_dense_torch  $seed $augmented 0.1
            train_dense_nerva  $seed $augmented 0.1
            train_sparse_torch $seed $augmented 0.1 0.05
            train_sparse_nerva $seed $augmented 0.1 0.05
        done
    done
}

#train_one
train_all
