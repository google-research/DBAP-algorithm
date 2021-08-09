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
data_dirs = ['/usr/local/google/home/abhishekunique/sim_franka/rpl_reset_free/recordings/play_targetrelabeled_MCS3.pkl',
            '/usr/local/google/home/abhishekunique/sim_franka/rpl_reset_free/recordings/play_targetrelabeled_MCS2.pkl',
            '/usr/local/google/home/abhishekunique/sim_franka/rpl_reset_free/recordings/play_targetlabeled_MCS1.pkl',
            '/usr/local/google/home/abhishekunique/sim_franka/rpl_reset_free/recordings/play_targetlabeled_MCS0.pkl',]

o_size = 13
all_demos = []
adjacency_matrix = np.zeros((8,8))
labeled_goals = [[] for _ in range(8)]
labeled_demos = [[] for _ in range(8)]

for di, data_dir in enumerate(data_dirs):
    demos = pickle.load(open(data_dir, 'rb'))
    demos = och_2_awac(demos)

    segment_idxs = []

    range_objs = [0.1, 0.1, 0.05, 0.05]
    for demo in demos:
        max_objs = np.array([0.25, 1, 0.75, -0.05])
        min_objs = np.array([0.1, 0.1, 0.15, -0.2])
        curr_segment_idx = []
        init_pos = demo['observations'][0, 2:6]
        init_bitflip = np.array([0, 0, 0, 1])
        for j in range(4):
            if init_pos[j] > max_objs[j]:
                init_bitflip[j] = 1
            elif init_pos[j] < min_objs[j]:
                init_bitflip[j] = 0
        final_pos = demo['observations'][-1, 2:6]
        final_bitflip = init_bitflip.copy()
        for j in range(4):
            if final_pos[j] > max_objs[j]:
                final_bitflip[j] = 1
            elif final_pos[j] < min_objs[j]:
                final_bitflip[j] = 0

        old_idx = 4*init_bitflip[0] + 2*init_bitflip[2] + init_bitflip[3]
        new_idx = 4 * final_bitflip[0] + 2 * final_bitflip[2] + final_bitflip[3]
        labeled_goals[int(new_idx)].append(demo['observations'][-1, :13])
        adjacency_matrix[int(old_idx), int(new_idx)] += 1

        goal = demo['observations'][-1, :o_size].copy()
        demo['observations'][:, o_size:] = goal.copy()
        demo['next_observations'][:, o_size:] = goal.copy()
        all_demos.append(demo)
        labeled_demos[int(old_idx)].append(demo)


    # import gym
    # import adept_envs
    # env = gym.make("franka_microwave_cabinet_slider-v1")
    # for demo in demos:
    #     qp_reset = demo['observations'][-1]
    #     mocap_reset = demo['observations'][-1, 6:9]
    #     env.sim.data.qpos[7:9] = qp_reset[:2].copy()
    #     env.sim.data.qpos[9:13] = qp_reset[2:6].copy()
    #     env.sim.data.mocap_pos[:] = mocap_reset.copy()
    #     for _ in range(100):
    #         env.sim.step()
    #     env.render()
    #     time.sleep(2)

# plt.imshow(adjacency_matrix)
# locs, labels = plt.xticks()
# label_list = itertools.product(['SC', 'SO'], ['CC','CO'], ['MO', 'MC'])
# label_list = ['-'.join(s) for s in label_list]
# plt.yticks(np.arange(8), label_list)
# plt.xticks(np.arange(8), label_list, rotation=90)
# plt.show()
for i in range(8):
    pickle.dump(labeled_demos[i], open('segmentedcollectedplay_microwave_cabinent_slider_perstart_%d.pkl'%i, 'wb'))
pickle.dump(all_demos, open('segmentedcollectedplay_microwave_cabinent_slider.pkl', 'wb'))