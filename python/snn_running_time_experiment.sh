#!/bin/bash

#--- fixed settings ---#
epochs=1
batchsize=100
momentum=0.9
lr=0.1
density=0.01
datadir="./data"

#--- variable settings ---#
seed=1
sizes="1024,512"
layers="ReLU;ReLU;Linear"
weights=xxx

# do one experiment with pytorch and nerva
function run
{
  sizes=$1
  layers=$2
  weights=$3
  logfile="snn/running-time/torch-density-$density-sizes-$sizes-seed-$seed.log"

  echo "Creating $logfile"
  python3 snn_training.py --torch \
                 --seed=$seed \
                 --overall-density=$density \
                 --lr=$lr \
                 --sizes="3072,$sizes,10" \
                 --batch-size=$batchsize \
                 --epochs=$epochs \
                 --momentum=$momentum \
                 --nesterov \
                 --datadir="$datadir" \
                 --custom-masking \
                 2>&1 | tee $logfile

  logfile="snn/running-time/nerva-density-$density-sizes-$sizes-seed-$seed.log"

  echo "Creating $logfile"
  ../tools/dist/mlp --seed=$seed \
                     --overall-density=$density \
                     --sizes="3072,$sizes,10" \
                     --batch-size=$batchsize \
                     --epochs=$epochs \
                     --learning-rate="multistep_lr($lr;50,75;0.1)" \
                     --optimizer="nesterov($momentum)" \
                     --layers="$layers" \
                     --weights=$weights \
                     --dataset=cifar10 \
                     --size=50000 \
                     --loss="softmax-cross-entropy" \
                     --threads=4 \
                     --no-shuffle \
                     -v \
                     2>&1 | tee $logfile
}

function run_all()
{
    for seed in 1
    do
        run "1024"                                              "ReLU;Linear"                                              "xx"
        run "1024,1024"                                         "ReLU;ReLU;Linear"                                         "xxx"
        run "1024,1024,1024"                                    "ReLU;eLU;ReLU;ReLU;Linear"                                "xxxx"
        run "1024,1024,1024,1024"                               "ReLU;ReLU;ReLU;ReLU;Linear"                               "xxxxx"
        run "1024,1024,1024,1024,1024"                          "ReLU;ReLU;ReLU;ReLU;ReLU;Linear"                          "xxxxxx"
        run "1024,1024,1024,1024,1024,1024"                     "ReLU;ReLU;ReLU;ReLU;ReLU;ReLU;Linear"                     "xxxxxxx"
        run "1024,1024,1024,1024,1024,1024,1024"                "ReLU;ReLU;ReLU;ReLU;ReLU;ReLU;ReLU;Linear"                "xxxxxxxx"
        run "1024,1024,1024,1024,1024,1024,1024,1024"           "ReLU;ReLU;ReLU;ReLU;ReLU;ReLU;ReLU;ReLU;Linear"           "xxxxxxxxx"
        run "1024,1024,1024,1024,1024,1024,1024,1024,1024"      "ReLU;ReLU;ReLU;ReLU;ReLU;ReLU;ReLU;ReLU;ReLU;Linear"      "xxxxxxxxxx"
        run "1024,1024,1024,1024,1024,1024,1024,1024,1024,1024" "ReLU;ReLU;ReLU;ReLU;ReLU;ReLU;ReLU;ReLU;ReLU;ReLU;Linear" "xxxxxxxxxxx"
        run "2048,2048,2048"                                    "ReLU;ReLU;ReLU;Linear"                                    "xxxx"
        run "4096,4096,4096"                                    "ReLU;ReLU;ReLU;Linear"                                    "xxxx"
        run "8192,8192,8192"                                    "ReLU;ReLU;ReLU;Linear"                                    "xxxx"
        run "16384,16384,16384"                                 "ReLU;ReLU;ReLU;Linear"                                    "xxxx"
        run "32768,32768,32768"                                 "ReLU;ReLU;ReLU;Linear"                                    "xxxx"
    done
}

run_all
