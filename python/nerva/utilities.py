# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from nervalib import RandomNumberGenerator, set_num_threads, global_timer_enable, global_timer_disable, \
    global_timer_suspend, global_timer_resume, global_timer_reset, global_timer_display


import time


class StopWatch(object):
    def __init__(self):
        self.start = time.perf_counter()

    def seconds(self):
        end = time.perf_counter()
        return end - self.start

    def reset(self):
        self.start = time.perf_counter()


class MapTimer:
    """
    Timer class with a map interface. For each key a start and stop value is stored.
    """

    def __init__(self):
        self.values = {}

    def start(self, key):
        self.values[key] = (time.perf_counter(), None)

    def stop(self, key):
        t = time.perf_counter()
        self.values[key] = (self.values[key][0], t)

    def milliseconds(self, key):
        t1, t2 = self.values[key]
        return (t2 - t1) * 1000

    def seconds(self, key):
        t1, t2 = self.values[key]
        return t2 - t1
