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

from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer

from rlkit.util.io import load_local_or_remote_file

import random
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.data_management.path_builder import PathBuilder

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

from rlkit.core import logger

import glob

class PathLoader:
    """
    Loads demonstrations and/or off-policy data into a Trainer
    """

    def load_demos(self, ):
        pass
