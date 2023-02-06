#!/bin/bash

epochs=100
batchsize=100
momentum=0.9

densesizes="1024,512"
densearchitecture=RRL
denseweights=XXX

sparsesizes="1024,512"
sparsearchitecture=RRL
sparseweights=XXX

datadir="./data"

function train_sparse_torch()
{
  local seed=$1
  local lr=$2
  local density=$3

  logfile="snn/torch-sparse-$density-augmented-seed$seed.log"

  python3 -u snn.py --torch --seed=$seed \
                    --density=$density \
                    --lr=$lr --sizes="3072,$sparsesizes,10" \
                    --batch-size=$batchsize \
                    --epochs=$epochs \
                    --momentum=$momentum \
                    --nesterov \
                    --custom-masking \
                    --datadir="$datadir" \
                    --preprocessed=./cifar$seed \
                    --precision=8 \
                    #--debug \
                    --export-weights-npz="weights-$density.npz" \
                    --custom-masking \
                    2>&1 | tee $logfile
}

function train_sparse_nerva()
{
  local seed=$1
  local lr=$2
  local density=$3

  logfile="snn/nerva-sparse-$density-augmented-seed$seed.log"

  ../tools/dist/mlpf --seed=$seed \
                     --density=$density \
                     --hidden="$sparsesizes" \
                     --batch-size=$batchsize \
                     --epochs=$epochs \
                     --learning-rate="multistep_lr($lr;50,75;0.1)" \
                     --optimizer="nesterov($momentum)" \
                     --architecture=$sparsearchitecture \
                     --weights=$sparseweights \
                     --dataset=cifar10 --size=50000 \
                     --loss="softmax-cross-entropy" \
                     --algorithm=minibatch \
                     --threads=4 \
                     --no-shuffle \
                     --verbose \
                     --preprocessed=./cifar$seed \
                     #--debug \
                     #--import-weights-npz="weights-$density.npz" \
                     2>&1 | tee $logfile
}

function train_dense_torch()
{
  local seed=$1
  local lr=$2

  logfile="snn/torch-dense-augmented-seed$seed.log"

  python3 -u snn.py --torch \
                    --seed=$seed \
                    --lr=$lr \
                    --sizes="3072,$densesizes,10" \
                    --batch-size=$batchsize \
                    --epochs=$epochs \
                    --momentum=$momentum \
                    --nesterov \
                    --custom-masking \
                    --datadir="$datadir" \
                    --preprocessed=./cifar$seed \
                    2>&1 | tee $logfile
}

function train_dense_nerva()
{
  local seed=$1
  local lr=$2

  logfile="snn/nerva-dense-augmented-seed$seed.log"

  ../tools/dist/mlpf --seed=$seed \
                     --hidden="$densesizes" \
                     --batch-size=$batchsize \
                     --epochs=$epochs \
                     --learning-rate="multistep_lr($lr;50,75;0.1)" \
                     --optimizer="nesterov($momentum)"  \
                     --architecture=$densearchitecture \
                     --weights=$denseweights \
                     --dataset=cifar10 \
                     --size=50000 \
                     --loss="softmax-cross-entropy" \
                     --algorithm=minibatch \
                     --threads=4 \
                     --verbose \
                     --preprocessed=./cifar$seed \
                     2>&1 | tee $logfile
}

function train_all()
{
    for seed in 1 2 3 4 5
    do
        train_sparse_torch $seed 0.1  0.001
        train_sparse_nerva $seed 0.1  0.001

        train_sparse_torch $seed 0.1  0.005
        train_sparse_nerva $seed 0.1  0.005

        train_sparse_torch $seed 0.1  0.01
        train_sparse_nerva $seed 0.1  0.01

        train_sparse_torch $seed 0.03 0.05
        train_sparse_nerva $seed 0.03 0.05

        train_sparse_torch $seed 0.03 0.1
        train_sparse_nerva $seed 0.03 0.1

        train_sparse_torch $seed 0.01 0.2
        train_sparse_nerva $seed 0.01 0.2

        train_sparse_torch $seed 0.01 0.5
        train_sparse_nerva $seed 0.01 0.5

        train_dense_torch  $seed 0.01
        train_dense_nerva  $seed 0.01
    done
}

train_all
