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

from rlkit.envs.proxy_env import ProxyEnv


class RewardWrapperEnv(ProxyEnv):
    """Substitute a different reward function"""

    def __init__(
            self,
            env,
            compute_reward_fn,
    ):
        ProxyEnv.__init__(self, env)
        self.spec = env.spec # hack for hand envs
        self.compute_reward_fn = compute_reward_fn

    def step(self, action):
        next_obs, reward, done, info = self._wrapped_env.step(action)
        info["env_reward"] = reward
        reward = self.compute_reward_fn(next_obs, reward, done, info)
        return next_obs, reward, done, info
