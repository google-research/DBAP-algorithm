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

import abc

from rlkit.policies.base import ExplorationPolicy


class ExplorationStrategy(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_action(self, t, observation, policy, **kwargs):
        pass

    def reset(self):
        pass


class RawExplorationStrategy(ExplorationStrategy, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_action_from_raw_action(self, action, **kwargs):
        pass

    def get_action(self, t, policy, *args, **kwargs):
        action, agent_info = policy.get_action(*args, **kwargs)
        return self.get_action_from_raw_action(action, t=t), agent_info

    def reset(self):
        pass


class PolicyWrappedWithExplorationStrategy(ExplorationPolicy):
    def __init__(
            self,
            exploration_strategy: ExplorationStrategy,
            policy,
    ):
        self.es = exploration_strategy
        self.policy = policy
        self.t = 0

    def set_num_steps_total(self, t):
        self.t = t

    def get_action(self, *args, **kwargs):
        return self.es.get_action(self.t, self.policy, *args, **kwargs)

    def reset(self):
        self.es.reset()
        self.policy.reset()
