#!/bin/bash
grep "epoch   1" *.log > results
python3 process_running_time_logs.py
