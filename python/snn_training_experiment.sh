#!/bin/bash
source utilities.sh

epochs=100
batchsize=100
momentum=0.9
trim_relu=0
weights=Xavier
loss=SoftmaxCrossEntropy
optimizers="Nesterov(0.9)"

#--- original experiment with 2 hidden layers ---#
#sizes="3072,1024,512,10"
#layers="ReLU;ReLU;Linear"

#--- larger experiment with 3 hidden layers ---#
#sizes="3072,1024,1024,1024,10"
#layers="ReLU;ReLU;ReLU;Linear"

#--- larger experiment with 3 hidden layers and trimmed ReLU ---#
sizes="3072,1024,1024,1024,10"
layers="TReLU(1e-30);TReLU(1e-30);TReLU(1e-30);Linear"
trim_relu="1e-30"

function train_sparse()
{
  local seed=$1
  local framework=$2
  local lr=$3
  local density=$4

  logfile="snn/training/$framework-sparse-$density-augmented-seed$seed.log"
  weightsfile="snn/training/weights-seed=$seed-density=$density.npz"

  # skip the run if a complete log already exists
  if [ -f "./$logfile" ] && ( grep -E -q "epoch\s+$epochs " "./$logfile" );
  then
      echo "A completed run in $logfile already exists"
      return 0
  fi

  print_header "Creating $logfile"

  if [ $framework == 'torch' ]; then
      python3 -u snn_training.py --torch \
                        --seed=$seed \
                        --overall-density=$density \
                        --lr=$lr --sizes="$sizes" \
                        --batch-size=$batchsize \
                        --epochs=$epochs \
                        --momentum=$momentum \
                        --nesterov \
                        --preprocessed=./cifar$seed \
                        --precision=8 \
                        --save-weights=$weightsfile \
                        --trim-relu=$trim_relu \
                        2>&1 | tee $logfile
  elif [ $framework == 'nerva' ]; then
      python3 -u snn_training.py --nerva \
                        --seed=$seed \
                        --overall-density=$density \
                        --lr=$lr --sizes="$sizes" \
                        --batch-size=$batchsize \
                        --epochs=$epochs \
                        --momentum=$momentum \
                        --nesterov \
                        --preprocessed=./cifar$seed \
                        --precision=8 \
                        --load-weights=$weightsfile \
                        --trim-relu=$trim_relu \
                        2>&1 | tee $logfile
  elif [ $framework == 'nervacpp' ]; then
      ../tools/dist/mlp --seed=$seed \
                        --overall-density=$density \
                        --sizes="$sizes" \
                        --batch-size=$batchsize \
                        --epochs=$epochs \
                        --learning-rate="MultistepLR($lr;50,75;0.1)" \
                        --optimizers=$optimizers \
                        --layers="$layers" \
                        --init-weights=$weights \
                        --loss=$loss \
                        --threads=4 \
                        --no-shuffle \
                        --verbose \
                        --preprocessed=./cifar$seed \
                        --load-weights=$weightsfile \
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
  weightsfile="snn/training/weights-seed=$seed.npz"

  # skip the run if a complete log already exists
  if [ -f "./$logfile" ] && ( grep -E -q "epoch\s+$epochs " "./$logfile" );
  then
      echo "A completed run in $logfile already exists"
      return 0
  fi

  print_header "Creating $logfile"

  if [ $framework == 'torch' ]; then
      python3 -u snn_training.py --torch \
                        --seed=$seed \
                        --lr=$lr --sizes="$sizes" \
                        --batch-size=$batchsize \
                        --epochs=$epochs \
                        --momentum=$momentum \
                        --nesterov \
                        --preprocessed=./cifar$seed \
                        --precision=8 \
                        --save-weights="$weightsfile" \
                        --trim-relu=$trim_relu \
                        2>&1 | tee $logfile
  elif [ $framework == 'nerva' ]; then
      python3 -u snn_training.py --nerva \
                        --seed=$seed \
                        --lr=$lr --sizes="$sizes" \
                        --batch-size=$batchsize \
                        --epochs=$epochs \
                        --momentum=$momentum \
                        --nesterov \
                        --preprocessed=./cifar$seed \
                        --precision=8 \
                        --load-weights="$weightsfile" \
                        2>&1 | tee $logfile
  elif [ $framework == 'nervacpp' ]; then
      ../tools/dist/mlp --seed=$seed \
                        --sizes="$sizes" \
                        --batch-size=$batchsize \
                        --epochs=$epochs \
                        --learning-rate="MultistepLR($lr;50,75;0.1)" \
                        --optimizers=$optimizers \
                        --layers="$layers" \
                        --init-weights=$weights \
                        --loss=$loss \
                        --threads=4 \
                        --verbose \
                        --preprocessed=./cifar$seed \
                        --load-weights="$weightsfile" \
                        2>&1 | tee $logfile
  fi
  echo ''
}

function train_all()
{
    for seed in 1 2 3 4 5
    do
        for framework in torch nerva nervacpp  # N.B. the order matters due to load/save of weights!
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
