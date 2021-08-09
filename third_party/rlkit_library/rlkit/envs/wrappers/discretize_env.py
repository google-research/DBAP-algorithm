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

import itertools

import numpy as np
from gym import Env
from gym.spaces import Discrete

from rlkit.envs.proxy_env import ProxyEnv


class DiscretizeEnv(ProxyEnv, Env):
    def __init__(self, wrapped_env, num_bins):
        super().__init__(wrapped_env)
        low = self.wrapped_env.action_space.low
        high = self.wrapped_env.action_space.high
        action_ranges = [
            np.linspace(low[i], high[i], num_bins)
            for i in range(len(low))
        ]
        self.idx_to_continuous_action = [
            np.array(x) for x in itertools.product(*action_ranges)
        ]
        self.action_space = Discrete(len(self.idx_to_continuous_action))

    def step(self, action):
        continuous_action = self.idx_to_continuous_action[action]
        return super().step(continuous_action)


