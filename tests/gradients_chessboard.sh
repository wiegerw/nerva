#!/bin/bash
source gradients.sh

dataset=chessboard
sizes="2,64,64,2"
epochs=10

tool=../tools/dist/mlpcolwisedouble
run_all 2>&1 | tee logs/gradients_chess_mlpcolwisedouble.log

tool=../tools/dist/mlprowwisedouble
run_all 2>&1 | tee logs/gradients_chess_mlprowwisedouble.log
