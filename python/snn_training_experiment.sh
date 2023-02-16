#!/bin/bash

epochs=100
batchsize=100
momentum=0.9
sizes="3072,1024,512,10"
architecture=RRL
weights=XXX

function print_header()
{
  local title="$1"
  local line="====================================================================================="
  local padding="$(printf '%0.1s' ' '{1..80})" # create 80 spaces
  local title_padding_length=$(( (80 - ${#title}) / 2 ))
  local title_padding="${padding:0:title_padding_length}"
  echo "$line"
  echo "===$title_padding$title$title_padding==="
  echo "$line"
}

function train_sparse()
{
  local seed=$1
  local framework=$2
  local lr=$3
  local density=$4

  logfile="snn/training/$framework-sparse-$density-augmented-seed$seed.log"

  # skip the run if a complete log already exists
  if [ -f "./$logfile" ] && ( grep -E -q "epoch\s+$epochs " "./$logfile" );
  then
      echo "A completed run in $logfile already exists"
      return 0
  fi

  print_header "Creating $logfile"

  if [ $framework == 'torch' ]; then
      python3 -u snn.py --torch \
                        --seed=$seed \
                        --overall-density=$density \
                        --lr=$lr --sizes="$sizes" \
                        --batch-size=$batchsize \
                        --epochs=$epochs \
                        --momentum=$momentum \
                        --nesterov \
                        --preprocessed=./cifar$seed \
                        --precision=8 \
                        --export-weights-npz="weights-$density.npz" \
                        2>&1 | tee $logfile
  elif [ $framework == 'nerva' ]; then
      python3 -u snn.py --nerva \
                        --seed=$seed \
                        --overall-density=$density \
                        --lr=$lr --sizes="$sizes" \
                        --batch-size=$batchsize \
                        --epochs=$epochs \
                        --momentum=$momentum \
                        --nesterov \
                        --preprocessed=./cifar$seed \
                        --precision=8 \
                        --import-weights-npz="weights-$density.npz" \
                        2>&1 | tee $logfile
  elif [ $framework == 'nervacpp' ]; then
      ../tools/dist/mlpf --seed=$seed \
                         --overall-density=$density \
                         --sizes="$sizes" \
                         --batch-size=$batchsize \
                         --epochs=$epochs \
                         --learning-rate="multistep_lr($lr;50,75;0.1)" \
                         --optimizer="nesterov($momentum)" \
                         --architecture=$architecture \
                         --weights=$weights \
                         --loss="softmax-cross-entropy" \
                         --algorithm=minibatch \
                         --threads=4 \
                         --no-shuffle \
                         --verbose \
                         --preprocessed=./cifar$seed \
                         --import-weights-npz="weights-$density.npz" \
                         2>&1 | tee $logfile
  fi
  echo ''
}

function train_dense()
{
  local seed=$1
  local framework=$2
  local lr=$3
  local density="1.0"

  logfile="snn/training/$framework-dense-augmented-seed$seed.log"

  # skip the run if a complete log already exists
  if [ -f "./$logfile" ] && ( grep -E -q "epoch\s+$epochs " "./$logfile" );
  then
      echo "A completed run in $logfile already exists"
      return 0
  fi

  print_header "Creating $logfile"

  if [ $framework == 'torch' ]; then
      python3 -u snn.py --torch \
                        --seed=$seed \
                        --lr=$lr --sizes="$sizes" \
                        --batch-size=$batchsize \
                        --epochs=$epochs \
                        --momentum=$momentum \
                        --nesterov \
                        --preprocessed=./cifar$seed \
                        --precision=8 \
                        --export-weights-npz="weights-$density.npz" \
                        2>&1 | tee $logfile
  elif [ $framework == 'nerva' ]; then
      python3 -u snn.py --nerva \
                        --seed=$seed \
                        --lr=$lr --sizes="$sizes" \
                        --batch-size=$batchsize \
                        --epochs=$epochs \
                        --momentum=$momentum \
                        --nesterov \
                        --preprocessed=./cifar$seed \
                        --precision=8 \
                        --import-weights-npz="weights-$density.npz" \
                        2>&1 | tee $logfile
  elif [ $framework == 'nervacpp' ]; then
      ../tools/dist/mlpf --seed=$seed \
                         --sizes="$sizes" \
                         --batch-size=$batchsize \
                         --epochs=$epochs \
                         --learning-rate="multistep_lr($lr;50,75;0.1)" \
                         --optimizer="nesterov($momentum)" \
                         --architecture=$architecture \
                         --weights=$weights \
                         --loss="softmax-cross-entropy" \
                         --algorithm=minibatch \
                         --threads=4 \
                         --verbose \
                         --preprocessed=./cifar$seed \
                         --import-weights-npz="weights-$density.npz" \
                         2>&1 | tee $logfile
  fi
  echo ''
}

function train_all()
{
    for seed in 1 2 3 4 5
    do
        for framework in torch nerva nervacpp  # N.B. the order matters due to import/export of weights!
        do
            train_sparse $seed $framework 0.1  0.001
            train_sparse $seed $framework 0.1  0.005
            train_sparse $seed $framework 0.1  0.01
            train_sparse $seed $framework 0.03 0.05
            train_sparse $seed $framework 0.03 0.1
            train_sparse $seed $framework 0.01 0.2
            train_sparse $seed $framework 0.01 0.5
            train_dense  $seed $framework 0.01
        done
    done
}

train_all
