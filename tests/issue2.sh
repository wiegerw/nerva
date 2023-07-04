#!/bin/bash
source ../python/utilities.sh
PYTHONPATH=../python

# This experiment demonstrated failing gradient checks in the first few epochs when dropout is used.
#
# epoch   0 lr: 0.00100000  loss: 0.69683839  train accuracy: 0.46000000  test accuracy: 0.50000000 time: 0.00000000s
# Db1[3] =      0.0930189              0              0              0              0              0
# Db1[4] =     -0.0814721              0              0              0              0              0
# Db1[5] =     -0.0986882              0              0              0              0              0

print_header "Dropout gradient checking"
../tools/dist/mlpd --epochs=10 --layers="ReLU(dropout=0.3);ReLU(dropout=0.3);Linear(dropout=0.3)" --sizes=2,64,64,2 --dataset=chessboard --init-weights=Xavier --batch-size=5 "--learning-rate=Constant(0.001)" --optimizers=GradientDescent --size=50 --loss=SoftmaxCrossEntropy --normalize --threads=4 --verbose --no-shuffle --seed=12345 --gradient-step=0.00001
