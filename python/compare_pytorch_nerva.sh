#!/bin/bash

python3 compare_pytorch_nerva.py --batch-size=50 --epochs=1 --learning-rate=0.001 --precision=5 --seed=159 --show --run=pytorch >& compare_pytorch_nerva1.log
python3 compare_pytorch_nerva.py --batch-size=50 --epochs=1 --learning-rate=0.001 --precision=5 --seed=159 --show --run=nerva   >& compare_pytorch_nerva2.log
meld compare_pytorch_nerva1.log compare_pytorch_nerva2.log
