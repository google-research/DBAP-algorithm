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
import itertools
import pickle
import sys
import time
sys.path.append("..")
from demo_2_awac import och_2_awac
data_dirs = ['/usr/local/google/home/abhishekunique/sim_franka/rpl_reset_free/recordings/play_segmented_CS.pkl',]

o_size = 13
num_elems = 2
num_possibilities = 2**num_elems
all_demos = []
adjacency_matrix = np.zeros((num_possibilities, num_possibilities))
labeled_goals = [[] for _ in range(4)]
labeled_demos = [[[] for _ in range(4)] for _ in range(4)]
def check_idx(curr_pos):
    max_objs = np.array([0.2, 1, 0.7, -0.05])
    min_objs = np.array([0.1, 0.1, 0.15, -0.2])
    curr_bitflips = np.zeros(4)
    for j in range(4):
        if curr_pos[j] > max_objs[j]:
            curr_bitflips[j] = 1
        elif curr_pos[j] < min_objs[j]:
            curr_bitflips[j] = 0
    new_idx = 2 * curr_bitflips[0] + curr_bitflips[2]
    return new_idx

idxs = []
for di, data_dir in enumerate(data_dirs):
    demos = pickle.load(open(data_dir, 'rb'))
    demos = och_2_awac(demos)
    for demo in demos:
        start_idx = check_idx(demo['observations'][0, 2:6])
        end_idx = check_idx(demo['observations'][-1, 2:6])
        goal = demo['observations'][-1, :o_size].copy()
        demo['observations'][:, o_size:] = goal.copy()
        demo['next_observations'][: , o_size:] = goal.copy()
        demo['rewards'] = np.zeros_like(demo['rewards'])
        demo[-1] = 1.0
        if start_idx == end_idx:
            continue
        labeled_goals[int(end_idx)].append(demo['observations'][-1, :o_size])
        labeled_demos[int(start_idx)][int(end_idx)].append(demo)
        all_demos.append(demo)
        idxs.append(end_idx)
        adjacency_matrix[int(start_idx), int(end_idx)] += 1

plt.imshow(adjacency_matrix)
label_list = itertools.product(['SC', 'SO'], ['CC','CO'])
label_list = ['-'.join(s) for s in label_list]
plt.yticks(np.arange(4), label_list)
plt.xticks(np.arange(4), label_list, rotation=90)
plt.show()
#
for start_idx in range(4):
    for end_idx in range(4):
        if len(labeled_demos[start_idx][end_idx]) > 0:
            plt.clf()
            plt.cla()
            fig, ax = plt.subplots(2, 5)
            for d in labeled_demos[start_idx][end_idx]:
                for i in range(2):
                    for j in range(5):
                        ax[i][j].plot(d['observations'][:, i*5 + j])
            plt.show()

pickle.dump(labeled_demos[2][3], open('demos_2_3_cabinet_slider.pkl', 'wb'))
# pickle.dump(all_demos, open('sim_relabeled_slider_cabinet_demos.pkl', 'wb'))
# pickle.dump(adjacency_matrix, open('sim_slider_cabinet_adjacency_matrix.pkl', 'wb'))
# pickle.dump(labeled_goals, open('sim_slider_cabinet_labeled_goals.pkl', 'wb'))