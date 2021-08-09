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

import random
from rlkit.exploration_strategies.base import RawExplorationStrategy
import numpy as np


class GaussianAndEpsilonStrategy(RawExplorationStrategy):
    """
    With probability epsilon, take a completely random action.
    with probability 1-epsilon, add Gaussian noise to the action taken by a
    deterministic policy.
    """
    def __init__(self, action_space, epsilon, max_sigma=1.0, min_sigma=None,
                 decay_period=1000000):
        assert len(action_space.shape) == 1
        if min_sigma is None:
            min_sigma = max_sigma
        self._max_sigma = max_sigma
        self._epsilon = epsilon
        self._min_sigma = min_sigma
        self._decay_period = decay_period
        self._action_space = action_space

    def get_action_from_raw_action(self, action, t=None, **kwargs):
        if random.random() < self._epsilon:
            return self._action_space.sample()
        else:
            sigma = self._max_sigma - (self._max_sigma - self._min_sigma) * min(1.0, t * 1.0 / self._decay_period)
            return np.clip(
                action + np.random.normal(size=len(action)) * sigma,
                self._action_space.low,
                self._action_space.high,
                )
