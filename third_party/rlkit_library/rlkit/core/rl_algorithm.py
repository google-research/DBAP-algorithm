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
from collections import OrderedDict
import numpy as np
import gtimer as gt

from rlkit.core import logger, eval_util
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import DataCollector
from collections import deque, OrderedDict
from rlkit.samplers.rollout_functions import rollout

def _get_epoch_timings():
    times_itrs = gt.get_times().stamps.itrs
    times = OrderedDict()
    epoch_time = 0
    for key in sorted(times_itrs):
        time = times_itrs[key][-1]
        epoch_time += time
        times['time/{} (s)'.format(key)] = time
    times['time/epoch (s)'] = epoch_time
    times['time/total (s)'] = gt.get_times().total
    return times


class BaseRLAlgorithm(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: DataCollector,
            evaluation_data_collector: DataCollector,
            replay_buffer: ReplayBuffer,
    ):
        self.trainer = trainer
        self.expl_env = exploration_env
        self.eval_env = evaluation_env
        self.expl_data_collector = exploration_data_collector
        self.eval_data_collector = evaluation_data_collector
        self.replay_buffer = replay_buffer
        self._start_epoch = 0

        self.post_epoch_funcs = []

    def train(self, start_epoch=0):
        self._start_epoch = start_epoch
        self._train()

    def _train(self):
        """
        Train model.
        """
        raise NotImplementedError('_train must implemented by inherited class')

    def _end_epoch(self, epoch):
        snapshot = self._get_snapshot()

        del snapshot['evaluation/env']
        del snapshot['exploration/env']

        # Information for what things are being visited
        # snapshot['visitations'] = self.expl_env.wrapped_env.density
        # snapshot['commanded_matrix'] = self.expl_env.wrapped_env.measurement_commanded_tasks
        # snapshot['reached_matrix'] = self.expl_env.wrapped_env.measurement_reached_tasks
        # snapshot['density'] = self.expl_env.wrapped_env.density
        # snapshot['transition_prob'] = self.expl_env.wrapped_env.transition_prob
        # snapshot['edge_visitation'] = self.expl_env.wrapped_env.edge_visitation
        # snapshot['goal_matrix'] = self.expl_env.wrapped_env.goal_matrix

        logger.save_itr_params(epoch, snapshot)
        gt.stamp('saving')
        self._log_stats(epoch)

        self.expl_data_collector.end_epoch(epoch)
        self.eval_data_collector.end_epoch(epoch)
        self.replay_buffer.end_epoch(epoch)
        self.trainer.end_epoch(epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        for k, v in self.expl_data_collector.get_snapshot().items():
            snapshot['exploration/' + k] = v
        for k, v in self.eval_data_collector.get_snapshot().items():
            snapshot['evaluation/' + k] = v
        for k, v in self.replay_buffer.get_snapshot().items():
            snapshot['replay_buffer/' + k] = v
        return snapshot

    def eval_policy_goalreaching(self):
        stats = OrderedDict()
        
        max_tries = 6 
        full_paths = [] 
        full_state_sequences = [] 
        full_goals = [] 

        for eval_itr_idx in range(5):
            for start_idx in range(4): 
                curr_goal = 3 - start_idx
                if start_idx == curr_goal:
                    continue

                self.eval_env.wrapped_env.reset_objs = True
                curr_path = [] 
                curr_sequences = [] 
                curr_state = start_idx

                print("======STARTING IS %d"%curr_state) 
                print("======GOAL IS %d"%curr_goal) 
                full_goals.append(curr_goal) 

                self.eval_env.wrapped_env.commanded_start = curr_state 
                self.eval_env.reset() 
                self.eval_env.wrapped_env.reset_objs = False 
                num_tried = 0     
                curr_sequences.append(curr_state) 
                while curr_state != curr_goal and num_tried < max_tries: 
                    nextp = self.eval_env.wrapped_env.reach_particular_goal(curr_goal) 
                    print("Next commanded goal is" + str(nextp))
                    self.eval_env.wrapped_env.commanded_goal = nextp
                    p = rollout(self.eval_env, self.trainer.eval_policy, max_path_length = 200) 
                    curr_state = self.eval_env.wrapped_env.check_goal_completion(self.eval_env.wrapped_env.get_obs_dict()['obj_qp']) 
                    num_tried += 1
                    curr_path.append(p) 
                    curr_sequences.append(curr_state) 
                    print("Next reached state is" + str(curr_state))
                full_paths.append(curr_path) 
                full_state_sequences.append(curr_sequences)

                stats['%d_%d_%d_success'%(eval_itr_idx, start_idx, curr_goal)] = np.array(curr_sequences[-1] == curr_goal, dtype=np.float32)
                stats['%d_%d_%d_pathlength'%(eval_itr_idx, start_idx, curr_goal)] = len(curr_sequences)
                stats['%d_%d_%d_pathreturn'%(eval_itr_idx, start_idx, curr_goal)] = np.mean(np.array([p['rewards'].sum() for p in curr_path]))
        
        self.eval_env.wrapped_env.reset_objs = True

        successes = []
        path_lengths = []
        avg_path_returns = []
        for fp, fss, fg in zip(full_paths, full_state_sequences, full_goals):
            is_success = (fss[-1] == fg)
            successes.append(is_success)
            path_lengths.append(len(fss))
            avg_path_return = np.mean(np.array([p['rewards'].sum() for p in fp]))
            avg_path_returns.append(avg_path_return)

        stats['overall_successes_mean'] = np.mean(np.array(successes))
        stats['overall_successes_std'] = np.std(np.array(successes))
        stats['overall_path_lengths_mean'] = np.mean(np.array(path_lengths))
        stats['overall_path_lengths_std'] = np.std(np.array(path_lengths))
        stats['overall_path_return'] = np.mean(np.array(avg_path_returns))
        return stats


    def eval_policy_exhaustive(self):
        stats = OrderedDict()
        # gm = np.array([[0, 1, 1, 0],
        #                 [1, 0, 0, 1],
        #                 [1, 0, 0, 1],
        #                 [0, 1, 1, 0]])
        gm = self.eval_env.wrapped_env.goal_matrix
        self.eval_env.wrapped_env.reset_objs = True
        num_elems = len(gm)
        overall_rs = []
        overall_rs_val = []
        for i in range(num_elems): 
            for j in  range(num_elems): 
                if gm[i][j] > 0: 
                    ps = [] 
                    for k in range(3):
                        self.eval_env.wrapped_env.commanded_start = i 
                        self.eval_env.wrapped_env.commanded_goal = j 
                        p = rollout(self.eval_env, self.trainer.policy, max_path_length=200) 
                        ps.append(p) 
                    rs = np.array([path['rewards'].max() for path in ps])
                    rs_val = np.array([path['rewards'].sum() for path in ps])
                    stats['%d_%d_eval'%(i, j)] = np.mean(rs)
                    stats['%d_%d_eval_val'%(i, j)] = np.mean(rs_val)
                    overall_rs.append(np.mean(rs))
                    overall_rs_val.append(np.mean(rs_val))
        overall_rs = np.array(overall_rs)
        overall_rs_val = np.array(overall_rs_val)
        stats['overall_eval'] = np.mean(overall_rs)
        stats['overall_eval_val'] = np.mean(overall_rs_val)
        return stats

    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        """
        Replay Buffer
        """
        # Uncomment to get results plotted
        # logger.record_dict(self.eval_policy_exhaustive(), prefix='eval_exhaustivess/')
        # logger.record_dict(self.eval_policy_goalreaching(), prefix='eval_goalreaching/')

        logger.record_dict(
            self.replay_buffer.get_diagnostics(),
            prefix='replay_buffer/'
        )

        """
        Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')

        """
        Exploration
        """
        logger.record_dict(
            self.expl_data_collector.get_diagnostics(),
            prefix='exploration/'
        )
        expl_paths = self.expl_data_collector.get_epoch_paths()
        if hasattr(self.expl_env, 'get_diagnostics'):
            logger.record_dict(
                self.expl_env.get_diagnostics(expl_paths),
                prefix='exploration/',
            )
        logger.record_dict(
            eval_util.get_generic_path_information(expl_paths),
            prefix="exploration/",
        )
        """
        Evaluation
        """
        logger.record_dict(
            self.eval_data_collector.get_diagnostics(),
            prefix='evaluation/',
        )
        eval_paths = self.eval_data_collector.get_epoch_paths()
        if hasattr(self.eval_env, 'get_diagnostics'):
            logger.record_dict(
                self.eval_env.get_diagnostics(eval_paths),
                prefix='evaluation/',
            )
        logger.record_dict(
            eval_util.get_generic_path_information(eval_paths),
            prefix="evaluation/",
        )

        """
        Misc
        """
        gt.stamp('logging')
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass
