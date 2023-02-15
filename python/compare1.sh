#!/bin/bash

# compare dense models
python3 -u compare_models.py --seed=1 --lr=0.1 --sizes=3072,1024,512,10 --batch-size=100 --epochs=1 --datadir=./data --weight-info

# compare sparse models
# python3 -u compare_models.py --seed=1 --overall-density=0.1 --lr=0.1 --sizes=3072,1024,512,10 --batch-size=100 --epochs=1 --datadir=./data
