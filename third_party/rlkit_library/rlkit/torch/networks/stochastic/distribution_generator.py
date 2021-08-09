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

from torch import nn

from rlkit.torch.distributions import (
    Bernoulli,
    Beta,
    Distribution,
    Independent,
    GaussianMixture as GaussianMixtureDistribution,
    GaussianMixtureFull as GaussianMixtureFullDistribution,
    MultivariateDiagonalNormal,
    TanhNormal,
)
from rlkit.torch.networks.basic import MultiInputSequential


class DistributionGenerator(nn.Module, metaclass=abc.ABCMeta):
    def forward(self, *input, **kwarg) -> Distribution:
        raise NotImplementedError


class ModuleToDistributionGenerator(
    MultiInputSequential,
    DistributionGenerator,
    metaclass=abc.ABCMeta
):
    pass


class Beta(ModuleToDistributionGenerator):
    def forward(self, *input):
        alpha, beta = super().forward(*input)
        return Beta(alpha, beta)


class Gaussian(ModuleToDistributionGenerator):
    def __init__(self, module, std=None, reinterpreted_batch_ndims=1):
        super().__init__(module)
        self.std = std
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

    def forward(self, *input):
        if self.std:
            mean = super().forward(*input)
            std = self.std
        else:
            mean, log_std = super().forward(*input)
            std = log_std.exp()
        return MultivariateDiagonalNormal(
            mean, std, reinterpreted_batch_ndims=self.reinterpreted_batch_ndims)


class BernoulliGenerator(ModuleToDistributionGenerator):
    def forward(self, *input):
        probs = super().forward(*input)
        return Bernoulli(probs)


class IndependentGenerator(ModuleToDistributionGenerator):
    def __init__(self, *args, reinterpreted_batch_ndims=1):
        super().__init__(*args)
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

    def forward(self, *input):
        distribution = super().forward(*input)
        return Independent(
            distribution,
            reinterpreted_batch_ndims=self.reinterpreted_batch_ndims,
        )


class GaussianMixture(ModuleToDistributionGenerator):
    def forward(self, *input):
        mixture_means, mixture_stds, weights = super().forward(*input)
        return GaussianMixtureDistribution(mixture_means, mixture_stds, weights)


class GaussianMixtureFull(ModuleToDistributionGenerator):
    def forward(self, *input):
        mixture_means, mixture_stds, weights = super().forward(*input)
        return GaussianMixtureFullDistribution(mixture_means, mixture_stds, weights)


class TanhGaussian(ModuleToDistributionGenerator):
    def forward(self, *input):
        mean, log_std = super().forward(*input)
        std = log_std.exp()
        return TanhNormal(mean, std)
