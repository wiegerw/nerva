# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import nervalib
import numpy as np

class DataSet(nervalib.DataSetView):
    def __init__(self, x_train, y_train, x_test, y_test):
        super().__init__(np.asfortranarray(x_train.T),
                         np.asfortranarray(y_train.T),
                         np.asfortranarray(x_test.T),
                         np.asfortranarray(y_test.T))
        # store references to the original data to make sure it is not destroyed
        self.keep_alive = [x_train, y_train, x_test, y_test]
