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
import pickle
import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector
from rlkit.core import logger
from rlkit.samplers.rollout_functions import rollout
import os
import cv2
import numpy as np

class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training

    def evaluate_exhaustive(self):
        eval_paths = []
        goal_matrix = pickle.load(open('goal_matrix_6elements_onehot_uniformsim_1elem.pkl', 'rb'))
        oem = self.eval_env.wrapped_env._eval_mode
        self.eval_env.wrapped_env._eval_mode = True
        self.eval_env.wrapped_env.idx_completion = True
        for start_idx in range(goal_matrix.shape[0]):
            for end_idx in range(goal_matrix.shape[0]):
                if goal_matrix[start_idx][end_idx]:
                    print("Evaluating start %d end %d" % (start_idx, end_idx))
                    self.eval_env.wrapped_env.commanded_start = start_idx
                    self.eval_env.wrapped_env.commanded_goal = end_idx
                    ep = rollout(self.eval_env, self.trainer.eval_policy, max_path_length=200,
                                render=True, render_kwargs={'mode': 'rgb_array'})
                    eval_paths.append(ep)
        saved_path = os.path.join(logger._snapshot_dir, 'saved_eval_paths.pkl')
        saved_img_path = os.path.join(logger._snapshot_dir, 'saved_img.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(saved_img_path, fourcc, 20.0, (84,  84))
        for p in eval_paths:
            for img in p['imgs']:
                img = img.astype(dtype=np.uint8)
                out.write(img)
            del p['imgs']
        out.release()
        self.eval_env.wrapped_env.idx_completion = False
        self.eval_env.wrapped_env._eval_mode = oem
        pickle.dump(eval_paths, open(saved_path, 'wb'))

    def _train(self):
        # self.evaluate_exhaustive()
        import IPython
        IPython.embed()
        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)
        self.expl_env._reset_counter = 0
        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            # self.eval_data_collector._env.wrapped_env._eval_mode = True
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp('evaluation sampling')
            # self.eval_data_collector._env.wrapped_env._eval_mode = False
            for _ in range(self.num_train_loops_per_epoch):

                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                if hasattr(self.expl_env, 'stop_motion'):
                    self.expl_env.stop_motion()
                gt.stamp('exploration sampling', unique=False)
                self.trainer.path_process(new_expl_paths, self.replay_buffer)
                self.replay_buffer.add_paths(new_expl_paths)
                gt.stamp('data storing', unique=False)

                self.training_mode(True)
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size)
                    self.trainer.train(train_data)
                gt.stamp('training', unique=False)
                self.training_mode(False)
            self._end_epoch(epoch)
