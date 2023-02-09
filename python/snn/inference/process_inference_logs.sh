#!/bin/bash
dirname="./seed123"
results="seed123.log"
grep 'Average' $dirname/*.log > $results
python3 process_inference_logs.py $results
