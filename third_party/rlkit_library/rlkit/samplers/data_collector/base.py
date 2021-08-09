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


class DataCollector(object, metaclass=abc.ABCMeta):
    def end_epoch(self, epoch):
        pass

    def get_diagnostics(self):
        return {}

    def get_snapshot(self):
        return {}

    @abc.abstractmethod
    def get_epoch_paths(self):
        pass


class PathCollector(DataCollector, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        pass


class StepCollector(DataCollector, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def collect_new_steps(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        pass
