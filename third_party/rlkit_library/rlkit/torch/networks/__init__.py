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
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
from rlkit.torch.networks.basic import (
    Clamp, ConcatTuple, Detach, Flatten, FlattenEach, Split, Reshape,
)
from rlkit.torch.networks.cnn import BasicCNN, CNN, MergedCNN, CNNPolicy
from rlkit.torch.networks.dcnn import DCNN, TwoHeadDCNN
from rlkit.torch.networks.feat_point_mlp import FeatPointMlp
from rlkit.torch.networks.image_state import ImageStatePolicy, ImageStateQ
from rlkit.torch.networks.linear_transform import LinearTransform
from rlkit.torch.networks.normalization import LayerNorm
from rlkit.torch.networks.mlp import (
    Mlp, ConcatMlp, MlpPolicy, TanhMlpPolicy,
    MlpQf,
    MlpQfWithObsProcessor,
    ConcatMultiHeadedMlp,
)
from rlkit.torch.networks.pretrained_cnn import PretrainedCNN
from rlkit.torch.networks.two_headed_mlp import TwoHeadMlp

__all__ = [
    'Clamp',
    'ConcatMlp',
    'ConcatMultiHeadedMlp',
    'ConcatTuple',
    'BasicCNN',
    'CNN',
    'CNNPolicy',
    'DCNN',
    'Detach',
    'FeatPointMlp',
    'Flatten',
    'FlattenEach',
    'LayerNorm',
    'LinearTransform',
    'ImageStatePolicy',
    'ImageStateQ',
    'MergedCNN',
    'Mlp',
    'PretrainedCNN',
    'Reshape',
    'Split',
    'TwoHeadDCNN',
    'TwoHeadMlp',
]

