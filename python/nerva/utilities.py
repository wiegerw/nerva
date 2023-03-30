# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from nervalib import RandomNumberGenerator, set_num_threads, global_timer_enable, global_timer_disable, \
    global_timer_suspend, global_timer_resume, global_timer_reset, global_timer_display

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
