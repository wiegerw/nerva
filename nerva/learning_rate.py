# Copyright 2022 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import nervalib


class LearningRateScheduler(nervalib.learning_rate_scheduler):
    pass


class ConstantScheduler(nervalib.constant_scheduler):
    pass


class TimeBasedScheduler(nervalib.time_based_scheduler):
    pass


class StepBasedScheduler(nervalib.step_based_scheduler):
    pass


class ExponentialScheduler(nervalib.exponential_scheduler):
    pass
