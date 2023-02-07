#!/bin/bash

#--- fixed settings ---#
seed=12345
epochs=1
batchsize=100
momentum=0.9
lr=0.1
density=0.01
datadir="./data"

#--- variable settings ---#
sizes="1024,512"
architecture=RRL
weights=xxx

# do one experiment with pytorch and nerva
function run
{
  sizes=$1
  architecture=$2
  weights=$3
  logfile="snn/running-time/torch-density-$density-sizes-$sizes-seed-$seed.log"

  echo "Creating $logfile"
  python3 snn.py --torch \
                 --seed=$seed \
                 --density=$density \
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
  ../tools/dist/mlpf --seed=$seed \
                     --density=$density \
                     --hidden="$sizes" \
                     --batch-size=$batchsize \
                     --epochs=$epochs \
                     --learning-rate="multistep_lr($lr;50,75;0.1)" \
                     --optimizer="nesterov($momentum)" \
                     --architecture=$architecture \
                     --weights=$weights \
                     --dataset=cifar10 \
                     --size=50000 \
                     --loss="softmax-cross-entropy" \
                     --algorithm=minibatch \
                     --threads=4 \
                     --no-shuffle \
                     --datadir="$datadir" \
                     -v \
                     2>&1 | tee $logfile
}

function run_all()
{
    for value in 1 2 3 4 5
    do
        seed=$value
        run "1024"                                              "RL"          "xx"
        run "1024,1024"                                         "RRL"         "xxx"
        run "1024,1024,1024"                                    "RRRL"        "xxxx"
        run "1024,1024,1024,1024"                               "RRRRL"       "xxxxx"
        run "1024,1024,1024,1024,1024"                          "RRRRRL"      "xxxxxx"
        run "1024,1024,1024,1024,1024,1024"                     "RRRRRRL"     "xxxxxxx"
        run "1024,1024,1024,1024,1024,1024,1024"                "RRRRRRRL"    "xxxxxxxx"
        run "1024,1024,1024,1024,1024,1024,1024,1024"           "RRRRRRRRL"   "xxxxxxxxx"
        run "1024,1024,1024,1024,1024,1024,1024,1024,1024"      "RRRRRRRRRL"  "xxxxxxxxxx"
        run "1024,1024,1024,1024,1024,1024,1024,1024,1024,1024" "RRRRRRRRRRL" "xxxxxxxxxxx"
        run "2048,2048,2048"                                    "RRRL"        "xxxx"
        run "4096,4096,4096"                                    "RRRL"        "xxxx"
        run "8192,8192,8192"                                    "RRRL"        "xxxx"
        run "16384,16384,16384"                                 "RRRL"        "xxxx"
        run "32768,32768,32768"                                 "RRRL"        "xxxx"
    done
}

run_all
