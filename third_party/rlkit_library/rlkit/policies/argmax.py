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
Torch argmax policy
"""
import numpy as np
from torch import nn

import rlkit.torch.pytorch_util as ptu
from rlkit.policies.base import Policy


class ArgmaxDiscretePolicy(nn.Module, Policy):
    def __init__(self, qf):
        super().__init__()
        self.qf = qf

    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        obs = ptu.from_numpy(obs).float()
        q_values = self.qf(obs).squeeze(0)
        q_values_np = ptu.get_numpy(q_values)
        return q_values_np.argmax(), {}
