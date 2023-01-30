#!/bin/bash

python3 augment_data.py --epochs=100 --seed=1 --outputdir=cifar1
python3 augment_data.py --epochs=100 --seed=2 --outputdir=cifar2
python3 augment_data.py --epochs=100 --seed=3 --outputdir=cifar3
python3 augment_data.py --epochs=100 --seed=4 --outputdir=cifar4
python3 augment_data.py --epochs=100 --seed=5 --outputdir=cifar5
