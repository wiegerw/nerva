# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import tensorflow as tf

def to_one_hot(x: tf.Tensor, n_classes: int):
    return tf.one_hot(x, n_classes)
