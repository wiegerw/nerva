#!/bin/bash

dirname=snn/storage
density=0.01
sizes="1024,512"

if [ ! -d "$dirname" ]
then
    echo "Creating directory $dirname"
    mkdir ./$dirname
fi

density=1.0
filename="$dirname/density-$density-sizes-$sizes.npy"
python3 snn_training.py --overall-density=$density --sizes="3072,$sizes,10" --save-model-npy="$filename"

density=0.1
filename="$dirname/density-$density-sizes-$sizes.npy"
python3 snn_training.py --overall-density=$density --sizes="3072,$sizes,10" --save-model-npy="$filename"

density=0.01
filename="$dirname/density-$density-sizes-$sizes.npy"
python3 snn_training.py --overall-density=$density --sizes="3072,$sizes,10" --save-model-npy="$filename"

density=0.001
filename="$dirname/density-$density-sizes-$sizes.npy"
python3 snn_training.py --overall-density=$density --sizes="3072,$sizes,10" --save-model-npy="$filename"

ls -lh $dirname
