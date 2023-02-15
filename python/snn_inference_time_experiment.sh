#!/bin/bash

dirname=snn/inference
density=0.01
sizes="1024,512"

if [ ! -d "$dirname" ]
then
    echo "Creating directory $dirname"
    mkdir ./$dirname
fi

for seed in 1 2 3
do
    for batch_size in 1 100
    do
        for density in 1.0 0.5 0.2 0.1 0.05 0.01 0.005 0.001
        do
            logfilename="$dirname/inference-batch-size-$batch_size-density-$density-seed-$seed.log"
            python3 -u snn.py --inference --custom-masking --batch-size=$batch_size --overall-density=$density --sizes="3072,$sizes,10" 2>&1 | tee $logfilename
        done
    done
done
