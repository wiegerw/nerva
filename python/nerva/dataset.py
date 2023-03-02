# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import nervalib
import numpy as np

class DataSet(nervalib.DataSetView):
    def __init__(self, Xtrain, Ttrain, Xtest, Ttest):
        super().__init__(Xtrain.T, Ttrain.T, Xtest.T, Ttest.T)
        # store references to the original data to make sure it is not destroyed
        self.keep_alive = [Xtrain, Ttrain, Xtest, Ttest]
