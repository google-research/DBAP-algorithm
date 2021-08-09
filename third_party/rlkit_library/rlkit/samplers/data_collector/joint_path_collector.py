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
from typing import Dict

from rlkit.core.logging import add_prefix
from rlkit.samplers.data_collector import PathCollector


class JointPathCollector(PathCollector):
    def __init__(self, path_collectors: Dict[str, PathCollector]):
        self.path_collectors = path_collectors

    def collect_new_paths(self, max_path_length, num_steps,
                          discard_incomplete_paths):
        paths = []
        for collector in self.path_collectors.values():
            collector.collect_new_paths(
                max_path_length, num_steps, discard_incomplete_paths
            )
        return paths

    def end_epoch(self, epoch):
        for collector in self.path_collectors.values():
            collector.end_epoch(epoch)

    def get_diagnostics(self):
        diagnostics = OrderedDict()
        for name, collector in self.path_collectors.items():
            diagnostics.update(
                add_prefix(collector.get_diagnostics(), name, divider='/'),
            )
        return diagnostics

    def get_snapshot(self):
        snapshot = {}
        for name, collector in self.path_collectors.items():
            snapshot.update(
                add_prefix(collector.get_snapshot(), name, divider='/'),
            )
        return snapshot

    def get_epoch_paths(self):
        paths = {}
        for name, collector in self.path_collectors.items():
            paths[name] = collector.get_epoch_paths()
        return paths

