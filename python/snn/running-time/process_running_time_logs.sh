#!/bin/bash
dirname="./seed1"
results="seed1.log"
grep "epoch   1" $dirname/*.log > $results
python3 process_running_time_logs.py $results
