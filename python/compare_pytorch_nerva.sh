#!/bin/bash

function train()
{
  logfile1="$1-torch.log"
  logfile2="$1-nerva.log"
  shift
  extra_args=$*

  echo "python3 compare_pytorch_nerva.py --batch-size=50 --epochs=1 --learning-rate=0.01 --precision=5 --seed=159 --momentum=0.9 --nesterov --copy --torch $extra_args >& $logfile1"
  python3 compare_pytorch_nerva.py --batch-size=50 --epochs=1 --learning-rate=0.01 --precision=5 --seed=159 --momentum=0.9 --nesterov --copy --torch $extra_args >& $logfile1

  echo "python3 compare_pytorch_nerva.py --batch-size=50 --epochs=1 --learning-rate=0.01 --precision=5 --seed=159 --momentum=0.9 --nesterov --copy --nerva $extra_args >& $logfile2"
  python3 compare_pytorch_nerva.py --batch-size=50 --epochs=1 --learning-rate=0.01 --precision=5 --seed=159 --momentum=0.9 --nesterov --copy --nerva $extra_args >& $logfile2
}

train "compare"
train "compare-augmented" "--augmented"
train "compare-sparse" "--density=0.5"
