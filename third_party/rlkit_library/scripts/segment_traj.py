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
data_dirs = ['/usr/local/google/home/abhishekunique/sim_franka/rpl_reset_free/recordings/recording_microwave_cabinet_slider_0.pkl',
            '/usr/local/google/home/abhishekunique/sim_franka/rpl_reset_free/recordings/recording_microwave_cabinet_slider_1.pkl',
            '/usr/local/google/home/abhishekunique/sim_franka/rpl_reset_free/recordings/recording_microwave_cabinet_slider_2.pkl',
            '/usr/local/google/home/abhishekunique/sim_franka/rpl_reset_free/recordings/recording_microwave_cabinet_slider_3.pkl',]

o_size = 13
all_demos = []
adjacency_matrix = np.zeros((4,4))
labeled_goals = [[] for _ in range(8)]
for di, data_dir in enumerate(data_dirs):
    demos = pickle.load(open(data_dir, 'rb'))
    demos = och_2_awac(demos)
    segment_idxs = []

    range_objs = [0.1, 0.1, 0.05, 0.05]
    for demo in demos:
        max_objs = np.array([0.2, 1, 0.7, -0.05])
        min_objs = np.array([0.1, 0.1, 0.15, -0.2])
        curr_segment_idx = []
        init_bitflips = np.array([0, 0, 0, 1])
        for i in range(len(demo['observations'])):
            curr_pos = demo['observations'][i, 9:13]
            curr_bitflips = init_bitflips.copy()
            for j in range(4):
                if curr_pos[j] > max_objs[j]:
                    curr_bitflips[j] = 1
                elif curr_pos[j] < min_objs[j]:
                    curr_bitflips[j] = 0
            if not np.all(curr_bitflips == init_bitflips):
                curr_segment_idx.append(i)
                old_idx = 2*init_bitflips[0] + init_bitflips[2]
                new_idx = 2*curr_bitflips[0] + curr_bitflips[2]
                labeled_goals[int(new_idx)].append(demo['observations'][i, :13])
                adjacency_matrix[int(old_idx), int(new_idx)] += 1
                init_bitflips = curr_bitflips.copy()
        segment_idxs.append(curr_segment_idx)
        # demo['observations'] = demo['observations'][:, 7:33]
        # demo['next_observations'] = demo['next_observations'][:, 7:33]

        # Relabeling according to segments
        start_idx = 0
        for j in range(len(curr_segment_idx)):
            goal = demo['observations'][curr_segment_idx[j], :o_size].copy()
            demo['observations'][start_idx:curr_segment_idx[j], o_size:] = goal.copy()
            demo['next_observations'][start_idx:curr_segment_idx[j], o_size:] = goal.copy()
            start_idx = curr_segment_idx[j]
        goal = demo['observations'][-1, :o_size].copy()
        demo['observations'][start_idx:, o_size:] = goal.copy()
        demo['next_observations'][start_idx:, o_size:] = goal.copy()
        all_demos.append(demo)


    # import gym
    # import adept_envs
    # env = gym.make("franka_microwave_cabinet_slider-v1")
    # for demo, si in zip(demos, segment_idxs):
    #     curr_idx = 0
    #     for i in range(len(demo['observations'])):
    #         print(i)
    #         qp_reset = demo['observations'][i, :13]
    #         mocap_reset = demo['observations'][i, 13:16]
    #         env.sim.data.qpos[:] = qp_reset
    #         env.sim.data.mocap_pos[:] = mocap_reset
    #         for _ in range(50):
    #             env.sim.step()
    #         if i == si[curr_idx]:
    #             env.render()
    #             time.sleep(0.1)
    #             curr_idx += 1
    #             if curr_idx == len(si) - 1:
    #                 break

#
# import gym
# import adept_envs
# env = gym.make("franka_microwave_cabinet_slider-v1")
# for lg in labeled_goals[3]:
#     qp_reset = lg[:13]
#     mocap_reset = lg[13:16]
#     env.sim.data.qpos[:] = qp_reset
#     env.sim.data.mocap_pos[:] = mocap_reset
#     for _ in range(50):
#         env.sim.step()
#     env.render()
#     time.sleep(0.1)
#
# plt.imshow(adjacency_matrix)
# locs, labels = plt.xticks()
# label_list = itertools.product(['SC', 'SO'], ['CC','CO'], ['MO', 'MC'])
# label_list = ['-'.join(s) for s in label_list]
# plt.yticks(np.arange(8), label_list)
# plt.xticks(np.arange(8), label_list, rotation=90)
# plt.show()

# pickle.dump(all_demos, open('segment_relabeled_microwave_cabinent_slider.pkl', 'wb'))
pickle.dump(adjacency_matrix, open('adjacency_matrix.pkl', 'wb'))
pickle.dump(labeled_goals, open('labeled_goals.pkl', 'wb'))