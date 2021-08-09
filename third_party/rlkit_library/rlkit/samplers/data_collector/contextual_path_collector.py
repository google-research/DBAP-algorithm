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

from functools import partial

from rlkit.envs.contextual import ContextualEnv
from rlkit.policies.base import Policy
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.samplers.rollout_functions import contextual_rollout


class ContextualPathCollector(MdpPathCollector):
    def __init__(
            self,
            env: ContextualEnv,
            policy: Policy,
            max_num_epoch_paths_saved=None,
            observation_key='observation',
            context_keys_for_policy='context',
            render=False,
            render_kwargs=None,
            **kwargs
    ):
        rollout_fn = partial(
            contextual_rollout,
            context_keys_for_policy=context_keys_for_policy,
            observation_key=observation_key,
        )
        super().__init__(
            env, policy, max_num_epoch_paths_saved, render, render_kwargs,
            rollout_fn=rollout_fn,
            **kwargs
        )
        self._observation_key = observation_key
        self._context_keys_for_policy = context_keys_for_policy

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot.update(
            observation_key=self._observation_key,
            context_keys_for_policy=self._context_keys_for_policy,
        )
        return snapshot
