#!/bin/bash

python3 snn.py --batch-size=50 --epochs=1 --lr=0.1 --precision=5 --seed=159 --copy --torch --nerva --show >& snn-test1.log
python3 snn.py --batch-size=50 --epochs=1 --lr=0.1 --precision=5 --seed=159 --copy --torch --nerva --show --density=0.1 >& snn-test2.log
python3 dense_sparse.py --batch-size=50 --epochs=1 --lr=0.1 --precision=5 --seed=159 --density=0.1 >& snn-test3.log
