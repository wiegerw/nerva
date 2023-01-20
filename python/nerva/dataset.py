# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import nervalib

class DataSet(nervalib.DataSetView):
    def __init__(self, x_train, y_train, x_test, y_test):
        super().__init__(x_train.T, y_train.T, x_test.T, y_test.T)  # N.B. the data needs to be transposed
        # store references to the original data to make sure it is not destroyed
        self.keep_alive = [x_train, y_train, x_test, y_test]
