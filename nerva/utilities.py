# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from nervalib import RandomNumberGenerator, set_num_threads

class StopWatch(object):
    def __init__(self):
        import time
        self.start = time.perf_counter()

    def seconds(self):
        import time
        end = time.perf_counter()
        return end - self.start

    def reset(self):
        import time
        self.start = time.perf_counter()
