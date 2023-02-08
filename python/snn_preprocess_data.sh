#!/bin/bash
#
# This script has been used to create preprocessed datasets for each
# epoch. These datasets contain augmented and shuffled data.
# Note that for the SNN experiments the datasets were created with
# batch size 50000. In the current version of the script the batch
# size is 100.

python3 snn_preprocess_data.py --epochs=100 --seed=1 --outputdir=cifar1
python3 snn_preprocess_data.py --epochs=100 --seed=2 --outputdir=cifar2
python3 snn_preprocess_data.py --epochs=100 --seed=3 --outputdir=cifar3
python3 snn_preprocess_data.py --epochs=100 --seed=4 --outputdir=cifar4
python3 snn_preprocess_data.py --epochs=100 --seed=5 --outputdir=cifar5
