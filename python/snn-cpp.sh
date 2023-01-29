#!/bin/bash

# create logfile directory
if [ ! -d snn ]; then
  mkdir snn
fi

../tools/dist/mlpf --epochs=100 --architecture=RRL --hidden="1024,512" --weights=xxx --dataset=cifar10 --learning-rate="multistep_lr(0.1;50,75;0.1)" --size=50000 --loss="softmax-cross-entropy" --algorithm=minibatch --batch-size=100 --normalize --threads=4 -v --no-shuffle --seed=1885661379 >& snn/dense-nerva-cpp.log

../tools/dist/mlpf --epochs=100 --architecture=RRL --hidden="1024,512" --weights=xxx --dataset=cifar10 --learning-rate="multistep_lr(0.1;50,75;0.1)" --size=50000 --loss="softmax-cross-entropy" --algorithm=minibatch --batch-size=100 --normalize --threads=4 -v --no-shuffle --seed=1885661379 --densities="0.041299715909090914,0.09292436079545456,1.0" >& snn/sparse-nerva-cpp.log
