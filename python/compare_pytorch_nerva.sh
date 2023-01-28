#!/bin/bash

function train()
{
  logfile1="$1-torch.log"
  logfile2="$1-nerva.log"
  shift
  extra_args=$*

  python3 compare_pytorch_nerva.py --batch-size=50 --epochs=1 --lr=0.1 --precision=5 --seed=159 --copy --torch $extra_args >& $logfile1
  python3 compare_pytorch_nerva.py --batch-size=50 --epochs=1 --lr=0.1 --precision=5 --seed=159 --copy --nerva $extra_args >& $logfile2

  meld $logfile1 $logfile2
}

train "c1" "--show"
train "c2" "--show --momentum=0.9"
train "c3" "--augmented"
train "c4" "--density=0.5"
