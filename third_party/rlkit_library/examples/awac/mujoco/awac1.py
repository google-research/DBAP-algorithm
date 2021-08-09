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

from rlkit.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
from rlkit.launchers.experiments.awac.awac_rl import experiment, process_args

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment

from rlkit.torch.sac.policies import GaussianPolicy
from rlkit.torch.networks import Clamp
import gym_point
import gym
import adept_envs
if __name__ == "__main__":
    variant = dict(
        num_epochs=501,
        num_eval_steps_per_epoch=3000,
        num_trains_per_train_loop=3000,
        num_expl_steps_per_train_loop=3000,
        min_num_steps_before_training=3000,
        max_path_length=300,
        batch_size=1024,
        replay_buffer_size=int(1E6),
        layer_size=256,
        num_layers=2,
        algorithm="AWAC",
        version="normal",
        collection_mode='batch',

        policy_class=GaussianPolicy,
        policy_kwargs=dict(
            hidden_sizes=[256] * 4,
            max_log_std=0,
            min_log_std=-6,
            std_architecture="values",
        ),
        qf_kwargs=dict(
            hidden_sizes=[256, 256]
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            beta=1,
            alpha=0,
            use_automatic_entropy_tuning=False,
            bc_num_pretrain_steps=100000,
            q_num_pretrain1_steps=0,
            q_num_pretrain2_steps=100000,
            policy_weight_decay=1e-4,
            train_bc_on_rl_buffer=False,
            buffer_policy_sample_actions=False,

            reparam_weight=0.0,
            awr_weight=1.0,
            bc_weight=0.0,
            compute_bc=False,
            awr_use_mle_for_vf=False,
            awr_sample_actions=False,
            awr_min_q=True,
        ),
        path_loader_kwargs=dict(
            demo_paths=[  # these can be loaded in awac_rl.py per env
                dict(
                    path='/usr/local/google/home/abhishekunique/sim_franka/rlkit/demos_slider_sim_play_awac_full_relabeled_window200_stride10.pkl',
                    obs_dict=False,
                    is_demo=True,
                    train_split=.9,
                ),
            ],
        ),
        path_loader_class=DictToMDPPathLoader,

        pretrain_rl=True,
        use_validation_buffer=True,
        add_env_demos=False,
        add_env_offpolicy_data=False,
        load_demos=True,
    )

    search_space = {
        'trainer_kwargs.beta':[.05, ],
        'train_rl':[True],
        'pretrain_rl':[True],
        'pretrain_policy':[True],
        'env_id': ['franka_slide-v1', ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )

    # n_seeds = 3
    # mode = 'gcp'
    # exp_prefix = 'skew-fit-pickup-reference-post-refactor'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=False,
                snapshot_gap=2,
                snapshot_mode='gap_and_last',
                num_exps_per_instance=3,
                gcp_kwargs=dict(
                    zone='us-west1-b',
                ),

            )

    # variants = []
    # for variant in sweeper.iterate_hyperparameters():
    #     variants.append(variant)

    # run_variants(experiment, variants, process_args)
