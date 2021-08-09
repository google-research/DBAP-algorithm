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
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.policies.base import ExplorationPolicy
from rlkit.torch.core import torch_ify, elem_or_tuple_to_numpy
from rlkit.torch.distributions import (
    Delta, TanhNormal, MultivariateDiagonalNormal, GaussianMixture, GaussianMixtureFull,
)
from rlkit.torch.networks import Mlp, CNN
from rlkit.torch.networks.basic import MultiInputSequential
from rlkit.torch.networks.stochastic.distribution_generator import (
    DistributionGenerator
)
from rlkit.torch.sac.policies.base import (
    TorchStochasticPolicy,
    PolicyFromDistributionGenerator,
    MakeDeterministic,
)


class PolicyFromQ(TorchStochasticPolicy):
    def __init__(
            self,
            qf,
            policy,
            num_samples=10,
            **kwargs
    ):
        super().__init__()
        self.qf = qf
        self.policy = policy
        self.num_samples = num_samples

    def forward(self, obs):
        with torch.no_grad():
            state = obs.repeat(self.num_samples, 1)
            action = self.policy(state).sample()
            q_values = self.qf(state, action)
            ind = q_values.max(0)[1]
        return Delta(action[ind])
