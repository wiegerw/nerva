#!/bin/bash
source gradients.sh

dataset=chessboard
sizes="2,64,64,2"
epochs=10

toolname=mlp_colwise_double
tool="../tools/dist/$toolname"
run_all 2>&1 | tee "logs/gradients_chess_$toolname.log"

toolname=mlp_rowwise_double
tool="../tools/dist/$toolname"
run_all 2>&1 | tee "logs/gradients_chess_$toolname.log"

toolname=mlp_colwise_mkl_double
tool="../tools/dist/$toolname"
run_all 2>&1 | tee "logs/gradients_chess_$toolname.log"

toolname=mlp_rowwise_mkl_double
tool="../tools/dist/$toolname"
run_all 2>&1 | tee "logs/gradients_chess_$toolname.log"
