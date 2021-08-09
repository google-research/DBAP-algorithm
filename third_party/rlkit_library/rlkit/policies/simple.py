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

from rlkit.policies.base import Policy


class RandomPolicy(Policy):
    """
    Policy that always outputs zero.
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, obs):
        return self.action_space.sample(), {}
