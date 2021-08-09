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

from collections import deque, OrderedDict
from functools import partial
import cv2
import numpy as np

from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.samplers.data_collector.base import PathCollector
from rlkit.samplers.rollout_functions import rollout
import rlkit.torch.pytorch_util as ptu
import matplotlib.pyplot as plt
import os
import torch
import skvideo
import skvideo.io
import matplotlib.pyplot as plt
from rlkit.core import logger

class MdpPathCollector(PathCollector):
    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
            rollout_fn=rollout,
            save_env_in_snapshot=True,
            goal_select=False,
            use_densities=False,
            replay_buffer=None,
            batch_size=100,
            algo=None,
            random_goal=False,
            name='expl'
    ):
        self._name = name
        self.random_goal = random_goal
        if render_kwargs is None:
            render_kwargs = {'mode': 'rgb_array'}
        self._env = env
        self._batch_size = batch_size
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs
        self._rollout_fn = rollout_fn
        self.goal_select = goal_select
        self._replay_buffer = replay_buffer
        self._algo = algo
        self._num_steps_total = 0
        self._num_paths_total = 0
        self._use_densities = use_densities
        self._densities = None
        self._save_env_in_snapshot = save_env_in_snapshot
        self.selected_goals = []

    def select_goal(self, batch, vfs, inv_densities):
        o_size = batch['observations'].shape[-1] // 2
        temperature = 1000.
        batch_score = np.exp((vfs + inv_densities) / temperature)  # UCB
        batch_score = batch_score / np.sum(batch_score)
        goal_idx = np.random.choice(np.arange(len(batch['observations'])), p=batch_score)
        return batch['observations'][goal_idx, :o_size]

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            # Start choosing random goals
            if self.random_goal and self._replay_buffer.num_steps_can_sample() > 0:
                print("SELECTING RANDOM GOAL")
                batch = self._replay_buffer.random_batch(self._batch_size)
                o_s = batch['observations'].copy()
                o_size = o_s.shape[-1] // 2
                idx_chosen = np.random.choice(range(len(o_s)))
                goal = o_s[idx_chosen][:o_size]
                self._env.set_goal(goal)
                self.selected_goals.append(goal)

            # Start choosing random goals via density + VF
            elif self.goal_select:
                os.makedirs('test_goal_select', exist_ok=True)

                batch = self._replay_buffer.random_batch(self._batch_size)
                o_s = batch['observations'].copy()
                o_size = o_s.shape[1] // 2
                o_s[:, o_size:] = o_s[:, :o_size].copy()
                curr_obs = self._env._get_obs().copy()
                o_s[:, :o_size] = curr_obs[:o_size]
                os_t = torch.from_numpy(o_s).float().to(ptu.device)
                vfs = self._algo.trainer.compute_vf(os_t).float().cpu().detach().numpy()
                inv_densities = self._algo.trainer.eval_rnd_inv_densities(batch)
                goal = self.select_goal(batch, vfs, inv_densities)

                self._env.set_goal(goal)
                self.selected_goals.append(goal)

            path = self._rollout_fn(
                self._env,
                self._policy,
                max_path_length=max_path_length_this_loop,
                render=self._render,
                render_kwargs=self._render_kwargs,
            )
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
            if self._use_densities:
                os.makedirs('test_densities', exist_ok=True)
                self._env.update_densities(path)
                self._densities = self._env.get_densities()
                plt.clf()
                plt.cla()
                plt.imshow(self._densities)
                plt.savefig('test_densities/densities.png')
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        plt.clf()
        plt.cla()
        fig, ax = plt.subplots(11, 1)
        for i in range(11):
            slider_pos = np.concatenate([p['observations'][:, i] for p in self._epoch_paths], axis=0)
            ax[i].plot(slider_pos)
        save_path = os.path.join(logger._snapshot_dir, 'slider_pos_%s_%d.png'%(self._name, epoch))
        plt.savefig(save_path)

        if self._render:
            imgs = np.concatenate([p['imgs'] for p in self._epoch_paths], axis=0)
            save_path = os.path.join(logger._snapshot_dir, 'vid_%s_%d.avi' % (self._name, epoch))
            # skvideo.io.vwrite(save_path, imgs)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(save_path, fourcc, 20.0, (84,  84))
            for img in imgs:
                img = img.astype(dtype=np.uint8)
                out.write(img)
            out.release()

        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        snapshot_dict = dict(
            policy=self._policy,
        )
        if self._save_env_in_snapshot:
            snapshot_dict['env'] = self._env
        return snapshot_dict


class GoalConditionedPathCollector(MdpPathCollector):
    def __init__(
            self,
            *args,
            observation_key='observation',
            desired_goal_key='desired_goal',
            goal_sampling_mode=None,
            **kwargs
    ):
        def obs_processor(o):
            return np.hstack((o[observation_key], o[desired_goal_key]))

        rollout_fn = partial(
            rollout,
            preprocess_obs_for_policy_fn=obs_processor,
        )
        super().__init__(*args, rollout_fn=rollout_fn, **kwargs)
        self._observation_key = observation_key
        self._desired_goal_key = desired_goal_key
        self._goal_sampling_mode = goal_sampling_mode

    def collect_new_paths(self, *args, **kwargs):
        self._env.goal_sampling_mode = self._goal_sampling_mode
        return super().collect_new_paths(*args, **kwargs)

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot.update(
            observation_key=self._observation_key,
            desired_goal_key=self._desired_goal_key,
        )
        return snapshot


class ObsDictPathCollector(MdpPathCollector):
    def __init__(
            self,
            *args,
            observation_key='observation',
            **kwargs
    ):
        def obs_processor(obs):
            return obs[observation_key]

        rollout_fn = partial(
            rollout,
            preprocess_obs_for_policy_fn=obs_processor,
        )
        super().__init__(*args, rollout_fn=rollout_fn, **kwargs)
        self._observation_key = observation_key

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot.update(
            observation_key=self._observation_key,
        )
        return snapshot


class VAEWrappedEnvPathCollector(GoalConditionedPathCollector):
    def __init__(
            self,
            env,
            policy,
            decode_goals=False,
            **kwargs
    ):
        """Expects env is VAEWrappedEnv"""
        super().__init__(env, policy, **kwargs)
        self._decode_goals = decode_goals

    def collect_new_paths(self, *args, **kwargs):
        self._env.decode_goals = self._decode_goals
        return super().collect_new_paths(*args, **kwargs)
