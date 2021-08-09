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

from rlkit.demos.source.demo_source import DemoSource
import pickle

from rlkit.data_management.path_builder import PathBuilder

from rlkit.util.io import load_local_or_remote_file

class HandDemoSource(DemoSource):
    def __init__(self, filename):
        self.data = load_local_or_remote_file(filename)

    def load_paths(self):
        paths = []
        for i in range(len(self.data)):
            p = self.data[i]
            H = len(p["observations"]) - 1

            path_builder = PathBuilder()

            for t in range(H):
                p["observations"][t]

                ob = path["observations"][t, :]
                action = path["actions"][t, :]
                reward = path["rewards"][t]
                next_ob = path["observations"][t+1, :]
                terminal = 0
                agent_info = {} # todo (need to unwrap each key)
                env_info = {} # todo (need to unwrap each key)

                path_builder.add_all(
                    observations=ob,
                    actions=action,
                    rewards=reward,
                    next_observations=next_ob,
                    terminals=terminal,
                    agent_infos=agent_info,
                    env_infos=env_info,
                )

            path = path_builder.get_all_stacked()
            paths.append(path)
        return paths
