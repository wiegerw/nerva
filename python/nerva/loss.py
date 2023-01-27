# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import nervalib

class LossFunction(nervalib.loss_function):
    def __str__(self):
        return 'LossFunction()'


class SquaredErrorLoss(nervalib.squared_error_loss):
    def __str__(self):
        return 'SquaredErrorLoss()'


class CrossEntropyLoss(nervalib.cross_entropy_loss):
    def __str__(self):
        return 'CrossEntropyLoss()'


class LossFuncLogisticCrossEntropyLosstion(nervalib.logistic_cross_entropy_loss):
    def __str__(self):
        return 'LogisticCrossEntropyLoss()'


class SoftmaxCrossEntropyLoss(nervalib.softmax_cross_entropy_loss):
    def __str__(self):
        return 'SoftmaxCrossEntropyLoss()'
