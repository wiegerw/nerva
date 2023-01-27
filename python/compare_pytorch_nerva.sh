#!/bin/bash

function train()
{
  logfile1=$1
  shift
  logfile2=$1
  shift
  extra_args=$*

  echo "python3 compare_pytorch_nerva.py --batch-size=50 --epochs=1 --learning-rate=0.01 --precision=5 --seed=159 --momentum=0.9 --nesterov --run=pytorch $extra_args >& $logfile1"
  python3 compare_pytorch_nerva.py --batch-size=50 --epochs=1 --learning-rate=0.01 --precision=5 --seed=159 --momentum=0.9 --nesterov --run=pytorch $extra_args >& $logfile1

  echo "python3 compare_pytorch_nerva.py --batch-size=50 --epochs=1 --learning-rate=0.01 --precision=5 --seed=159 --momentum=0.9 --nesterov --run=nerva $extra_args >& $logfile2"
  python3 compare_pytorch_nerva.py --batch-size=50 --epochs=1 --learning-rate=0.01 --precision=5 --seed=159 --momentum=0.9 --nesterov --run=nerva $extra_args >& $logfile2
}

train "compare1.log" "compare2.log"
train "compare3.log" "compare4.log" "--augmented"
train "compare5.log" "compare6.log" "--density=0.5"
