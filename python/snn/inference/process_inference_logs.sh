#!/bin/bash
grep "epoch   1" seed123/*.log > results
python3 process_inference_logs.py
