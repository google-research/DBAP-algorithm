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
import time
sys.path.append("..")
from demo_2_awac import och_2_awac
data_dirs = ['/usr/local/google/home/abhishekunique/sim_franka/rpl_reset_free/recordings/play_slider_0_sim.pkl',
             '/usr/local/google/home/abhishekunique/sim_franka/rpl_reset_free/recordings/play_slider_0_sim.pkl', ]

o_size = 10
all_demos = []
for di, data_dir in enumerate(data_dirs):
    demos = pickle.load(open(data_dir, 'rb'))
    demos = och_2_awac(demos)

    segment_idxs = []

    for demo in demos:
        max_objs = np.array([0.25])
        min_objs = np.array([0.1])
        curr_segment_idx = []
        init_bitflips = np.array([0])
        for i in range(len(demo['observations'])):
            curr_pos = demo['observations'][i, 9:10]
            curr_bitflips = init_bitflips.copy()
            for j in range(1):
                if curr_pos[j] > max_objs[j]:
                    curr_bitflips[j] = 1
                elif curr_pos[j] < min_objs[j]:
                    curr_bitflips[j] = 0
            if not np.all(curr_bitflips == init_bitflips):
                curr_segment_idx.append(i)
                init_bitflips = curr_bitflips.copy()
        segment_idxs.append(curr_segment_idx)
        demo['observations'] = demo['observations'][:, 7:17]
        demo['observations'] = np.concatenate([demo['observations'],
                                               np.zeros((len(demo['observations']), 10))], axis=1)
        demo['next_observations'] = demo['next_observations'][:, 7:17]
        demo['next_observations'] = np.concatenate([demo['next_observations'],
                                                    np.zeros((len(demo['next_observations']), 10))], axis=1)
        # Relabeling according to segments
        start_idx = 0
        for j in range(len(curr_segment_idx)):
            demo['observations'][start_idx:curr_segment_idx[j], o_size:] = \
                demo['observations'][curr_segment_idx[j], :o_size].copy()
            demo['next_observations'][start_idx:curr_segment_idx[j], o_size:] = \
                demo['observations'][curr_segment_idx[j], :o_size].copy()
            start_idx = curr_segment_idx[j]
        demo['observations'][start_idx:, o_size:] = demo['observations'][-1, :o_size]
        demo['next_observations'][start_idx:, o_size:] = demo['observations'][-1, :o_size]
        all_demos.append(demo)
    # demos = pickle.load(open(data_dir, 'rb'))
    # demos = och_2_awac(demos)
    # import gym
    # import adept_envs
    # env = gym.make("franka_slide-v1")
    # for demo, si in zip(demos, segment_idxs):
    #     curr_idx = 0
    #     for i in range(len(demo['observations'])):
    #         print(i)
    #         qp_reset = demo['observations'][i, :10]
    #         mocap_reset = demo['observations'][i, 10:13]
    #         env.sim.data.qpos[:] = qp_reset
    #         env.sim.data.mocap_pos[:] = mocap_reset
    #         for _ in range(50):
    #             env.sim.step()
    #         if i == si[curr_idx]:
    #             env.render()
    #             time.sleep(1)
    #             curr_idx += 1
    #             if curr_idx == len(si) - 1:
    #                 break
pickle.dump(all_demos, open('segment_relabeled_slider.pkl', 'wb'))