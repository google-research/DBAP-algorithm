# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
General utility functions for machine learning.
"""
import abc
import math
import numpy as np


class ScalarSchedule(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_value(self, t):
        pass


class ConstantSchedule(ScalarSchedule):
    def __init__(self, value):
        self._value = value

    def get_value(self, t):
        return self._value


class LinearSchedule(ScalarSchedule):
    """
    Linearly interpolate and then stop at a final value.
    """
    def __init__(
            self,
            init_value,
            final_value,
            ramp_duration,
    ):
        self._init_value = init_value
        self._final_value = final_value
        self._ramp_duration = ramp_duration

    def get_value(self, t):
        return (
            self._init_value
            + (self._final_value - self._init_value)
            * min(1.0, t * 1.0 / self._ramp_duration)
        )


class IntLinearSchedule(LinearSchedule):
    """
    Same as RampUpSchedule but round output to an int
    """
    def get_value(self, t):
        return int(super().get_value(t))


class PiecewiseLinearSchedule(ScalarSchedule):
    """
    Given a list of (x, t) value-time pairs, return value x at time t,
    and linearly interpolate between the two
    """
    def __init__(
            self,
            x_values,
            y_values,
    ):
        self._x_values = x_values
        self._y_values = y_values

    def get_value(self, t):
        return np.interp(t, self._x_values, self._y_values)


class IntPiecewiseLinearSchedule(PiecewiseLinearSchedule):
    def get_value(self, t):
        return int(super().get_value(t))


def none_to_infty(bounds):
    if bounds is None:
        bounds = -math.inf, math.inf
    lb, ub = bounds
    if lb is None:
        lb = -math.inf
    if ub is None:
        ub = math.inf
    return lb, ub
