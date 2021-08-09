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
from collections import OrderedDict


LossStatistics = OrderedDict


class LossFunction(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def compute_loss(self, batch, skip_statistics=False, **kwargs):
        """Returns loss and statistics given a batch of data.
        batch : Data to compute loss of
        skip_statistics: Whether statistics should be calculated. If True, then
            an empty dict is returned for the statistics.

        Returns: (loss, stats) tuple.
        """
        pass
