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

from collections import deque

import numpy as np
from gym import Env
from gym.spaces import Box

from rlkit.envs.proxy_env import ProxyEnv


class HistoryEnv(ProxyEnv, Env):
    def __init__(self, wrapped_env, history_len):
        super().__init__(wrapped_env)
        self.history_len = history_len

        high = np.inf * np.ones(
            self.history_len * self.observation_space.low.size)
        low = -high
        self.observation_space = Box(low=low,
                                     high=high,
                                     )
        self.history = deque(maxlen=self.history_len)

    def step(self, action):
        state, reward, done, info = super().step(action)
        self.history.append(state)
        flattened_history = self._get_history().flatten()
        return flattened_history, reward, done, info

    def reset(self, **kwargs):
        state = super().reset()
        self.history = deque(maxlen=self.history_len)
        self.history.append(state)
        flattened_history = self._get_history().flatten()
        return flattened_history

    def _get_history(self):
        observations = list(self.history)

        obs_count = len(observations)
        for _ in range(self.history_len - obs_count):
            dummy = np.zeros(self._wrapped_env.observation_space.low.size)
            observations.append(dummy)
        return np.c_[observations]


