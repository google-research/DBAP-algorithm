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


class EpsilonGreedy(RawExplorationStrategy):
    """
    Take a random discrete action with some probability.
    """
    def __init__(self, action_space, prob_random_action=0.1):
        self.prob_random_action = prob_random_action
        self.action_space = action_space

    def get_action_from_raw_action(self, action, **kwargs):
        if random.random() <= self.prob_random_action:
            return self.action_space.sample()
        return action
