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

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys

sys.path.append("..")
from demo_2_awac import och_2_awac

def end_relabel(path):
    o_size = path['observations'].shape[1] // 2
    path['observations'][:, o_size:] = path['observations'][-1, :o_size]
    path['next_observations'][:, o_size:] = path['observations'][-1, :o_size]
    rs = np.abs(path['observations'][:, 0] - path['observations'][-1, 0]) < 0.3
    rs = np.array(rs, dtype=np.float32)
    path['rewards'] = rs
    return path

bc_relabeled = []
paths = ['/usr/local/google/home/abhishekunique/hardware_franka/rpl_reset_free/recordings/demo_open _corrective_jan7.pkl']
for path in paths:
    data = pickle.load(open(path,'rb'))
    awac_formatted_list = och_2_awac(data)
    for l in awac_formatted_list:
        l['observations'] = np.concatenate([l['observations'][:, 1:2],
                                            l['observations'][:, 10:18],
                                            l['observations'][:, 25:34]], axis=1)
        l['next_observations'] = np.concatenate([l['next_observations'][:, 1:2],
                                                 l['next_observations'][:, 10:18],
                                                 l['next_observations'][:, 25:34]], axis=1)
    for dat in awac_formatted_list:
        relabeled_list = end_relabel(dat)
        bc_relabeled += [relabeled_list]
lens = sum(len(a['observations']) for a in bc_relabeled)
print(lens)
import IPython
IPython.embed()
output_path = '/usr/local/google/home/abhishekunique/hardware_franka/demo_open_corrective_Jan7_laststeprelabeled_cutobs.pkl'
pickle.dump(bc_relabeled, open(output_path,'wb'))