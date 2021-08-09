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

import gym
import os
os.environ['MUJOCO_GL'] = 'osmesa'
# import roboverse
# from rlkit.data_management.awr_env_replay_buffer import AWREnvReplayBuffer
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.data_management.split_buffer import SplitReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv, StackObservationEnv, RewardWrapperEnv
import rlkit.torch.pytorch_util as ptu
from rlkit.samplers.data_collector import MdpPathCollector, ObsDictPathCollector
from rlkit.samplers.data_collector.step_collector import MdpStepCollector
from rlkit.torch.networks import ConcatMlp, Mlp
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic, GaussianMixturePolicy, GaussianPolicy
from rlkit.torch.sac.awac_trainer import AWACTrainer
from rlkit.torch.torch_rl_algorithm import (
    TorchBatchRLAlgorithm,
    TorchOnlineRLAlgorithm,
)
import gym_point
import gym
import adept_envs

from rlkit.demos.source.hdf5_path_loader import HDF5PathLoader
from rlkit.demos.source.mdp_path_loader import MDPPathLoader
# from rlkit.visualization.video import save_paths, VideoSaveFunction

from multiworld.core.flat_goal_env import FlatGoalEnv
from multiworld.core.image_env import ImageEnv
from multiworld.core.gym_to_multi_env import GymToMultiEnv

import torch
import numpy as np
from torchvision.utils import save_image

from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_and_epsilon_strategy import GaussianAndEpsilonStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy

import os.path as osp
from rlkit.core import logger
from rlkit.util.io import load_local_or_remote_file
import pickle
from rlkit.samplers.rollout_functions import rollout

# from rlkit.envs.images import Renderer, InsertImageEnv, EnvRenderer
from rlkit.envs.make_env import make

ENV_PARAMS = {
    'HalfCheetah-v2': {
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_demo_path': dict(
            path="demos/icml2020/mujoco/hc_action_noise_15.npy",
            obs_dict=False,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            path="demos/icml2020/mujoco/hc_off_policy_15_demos_100.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'Ant-v2': {
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_demo_path': dict(
            path="demos/icml2020/mujoco/ant_action_noise_15.npy",
            obs_dict=False,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            path="demos/icml2020/mujoco/ant_off_policy_15_demos_100.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'Walker2d-v2': {
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'env_demo_path': dict(
            path="demos/icml2020/mujoco/walker_action_noise_15.npy",
            obs_dict=False,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            path="demos/icml2020/mujoco/walker_off_policy_15_demos_100.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },

    'SawyerRigGrasp-v0': {
        'env_id': 'SawyerRigGrasp-v0',
        # 'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 50,
        # 'num_epochs': 1000,
    },

    'pen-binary-v0': {
        'env_id': 'pen-binary-v0',
        'max_path_length': 200,
        'env_demo_path': dict(
            path="demos/icml2020/hand/pen2_sparse.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            path="demos/icml2020/hand/pen_bc_sparse4.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'door-binary-v0': {
        'env_id': 'door-binary-v0',
        'max_path_length': 200,
        'env_demo_path': dict(
            path="demos/icml2020/hand/door2_sparse.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            path="demos/icml2020/hand/door_bc_sparse4.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'relocate-binary-v0': {
        'env_id': 'relocate-binary-v0',
        'max_path_length': 200,
        'env_demo_path': dict(
            path="demos/icml2020/hand/relocate2_sparse.npy",
            obs_dict=True,
            is_demo=True,
        ),
        'env_offpolicy_data_path': dict(
            path="demos/icml2020/hand/relocate_bc_sparse4.npy",
            obs_dict=False,
            is_demo=False,
            train_split=0.9,
        ),
    },
    'PointPlayEnv-v0': {
        'env_id': 'PointPlayEnv-v0',
        'max_path_length': 100,
    },
    'franka_slide-v1': {
        'env_id': 'franka_slide-v1',
    },
    'franka_microwave_cabinet_slider_resetfree-v1': {
        'env_id': 'franka_microwave_cabinet_slider_resetfree-v1',
    },
    'SliderResetFree-v0': {
        'env_id': 'SliderResetFree-v0'
    },
    'SliderCabinetResetFree-v0': {
        'env_id': 'SliderCabinetResetFree-v0'
    },
    'SliderCabinetKnobResetFree-v0': {
        'env_id': 'SliderCabinetKnobResetFree-v0'
    },
    'SliderCabinetResetFreeNew-v0': {
        'env_id': 'SliderCabinetResetFreeNew-v0'
    }
}


def resume(variant):
    data = load_local_or_remote_file(variant.get("pretrained_algorithm_path"), map_location="cuda")
    return data


def process_args(variant):
    if variant.get("debug", False):
        variant['max_path_length'] = 50
        variant['batch_size'] = 5
        variant['num_epochs'] = 5
        # variant['num_eval_steps_per_epoch'] = 100
        # variant['num_expl_steps_per_train_loop'] = 100
        variant['num_trains_per_train_loop'] = 10
        # variant['min_num_steps_before_training'] = 100
        variant['trainer_kwargs']['bc_num_pretrain_steps'] = min(10, variant['trainer_kwargs'].get('bc_num_pretrain_steps', 0))
        variant['trainer_kwargs']['q_num_pretrain1_steps'] = min(10, variant['trainer_kwargs'].get('q_num_pretrain1_steps', 0))
        variant['trainer_kwargs']['q_num_pretrain2_steps'] = min(10, variant['trainer_kwargs'].get('q_num_pretrain2_steps', 0))

def experiment(variant):
    # if variant.get("pretrained_algorithm_path", False):
    #     resume(variant)
    #     return

    normalize_env = variant.get('normalize_env', True)
    env_id = variant.get('env_id', None)
    env_params = ENV_PARAMS.get(env_id, {})
    variant.update(env_params)
    env_class = variant.get('env_class', None)
    env_kwargs = variant.get('env_kwargs', {})
    eval_env_kwargs = variant.get('eval_env_kwargs', env_kwargs)
    expl_env = make(env_id, env_class, env_kwargs, normalize_env)
    if variant.get('expl_eval_same', True):
        eval_env = expl_env
    else:
        eval_env = make(env_id, env_class, eval_env_kwargs, normalize_env)

    if variant.get('add_env_demos', False):
        variant["path_loader_kwargs"]["demo_paths"].append(variant["env_demo_path"])
    if variant.get('add_env_offpolicy_data', False):
        variant["path_loader_kwargs"]["demo_paths"].append(variant["env_offpolicy_data_path"])

    path_loader_kwargs = variant.get("path_loader_kwargs", {})
    stack_obs = path_loader_kwargs.get("stack_obs", 1)
    if stack_obs > 1:
        expl_env = StackObservationEnv(expl_env, stack_obs=stack_obs)
        eval_env = StackObservationEnv(eval_env, stack_obs=stack_obs)

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    if hasattr(expl_env, 'info_sizes'):
        env_info_sizes = expl_env.info_sizes
    else:
        env_info_sizes = dict()

    qf_kwargs = variant.get("qf_kwargs", {})
    rnd_kwargs = variant.get("rnd_kwargs", qf_kwargs)
    rnd_size = variant.get("rnd_size", 32)
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **qf_kwargs
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **qf_kwargs
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **qf_kwargs
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **qf_kwargs
    )

    rnd_net1 = Mlp(
        input_size=obs_dim // 2,
        output_size=rnd_size,
        **rnd_kwargs
    )

    rnd_net2 = Mlp(
        input_size=obs_dim // 2,
        output_size=rnd_size,
        **rnd_kwargs
    )

    policy_class = variant.get("policy_class", TanhGaussianPolicy)
    if policy_class == 'GaussianMixturePolicy':
        policy_class = GaussianMixturePolicy
    elif policy_class == 'TanhGaussianPolicy':
        policy_class = TanhGaussianPolicy
    elif policy_class == 'GaussianPolicy':
        policy_class =  GaussianPolicy
    policy_kwargs = variant['policy_kwargs']
    policy_path = variant.get("policy_path", False)
    if policy_path:
        policy = load_local_or_remote_file(policy_path)
    else:
        policy = policy_class(
            obs_dim=obs_dim,
            action_dim=action_dim,
            **policy_kwargs,
        )
    goal_model = policy_class(
        obs_dim=int(obs_dim // 2),
        action_dim=int(obs_dim // 2),
        **policy_kwargs,
    )

    buffer_policy_path = variant.get("buffer_policy_path", False)
    if buffer_policy_path:
        buffer_policy = load_local_or_remote_file(buffer_policy_path)
    else:
        buffer_policy_class = variant.get("buffer_policy_class", policy_class)
        buffer_policy = buffer_policy_class(
            obs_dim=obs_dim,
            action_dim=action_dim,
            **variant.get("buffer_policy_kwargs", policy_kwargs),
        )

    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
        name='eval',
        render=variant.get('render_eval', False)
    )

    expl_policy = policy
    exploration_kwargs =  variant.get('exploration_kwargs', {})
    if exploration_kwargs:
        if exploration_kwargs.get("deterministic_exploration", False):
            expl_policy = MakeDeterministic(policy)

        exploration_strategy = exploration_kwargs.get("strategy", None)
        if exploration_strategy is None:
            pass
        elif exploration_strategy == 'ou':
            es = OUStrategy(
                action_space=expl_env.action_space,
                max_sigma=exploration_kwargs['noise'],
                min_sigma=exploration_kwargs['noise'],
            )
            expl_policy = PolicyWrappedWithExplorationStrategy(
                exploration_strategy=es,
                policy=expl_policy,
            )
        elif exploration_strategy == 'gauss_eps':
            es = GaussianAndEpsilonStrategy(
                action_space=expl_env.action_space,
                max_sigma=exploration_kwargs['noise'],
                min_sigma=exploration_kwargs['noise'],  # constant sigma
                epsilon=0,
            )
            expl_policy = PolicyWrappedWithExplorationStrategy(
                exploration_strategy=es,
                policy=expl_policy,
            )
        else:
            error

    main_replay_buffer_kwargs=dict(
        max_replay_buffer_size=variant['replay_buffer_size'],
        env=expl_env,
    )
    replay_buffer_kwargs = dict(
        max_replay_buffer_size=variant['replay_buffer_size'],
        env=expl_env,
    )

    replay_buffer = variant.get('replay_buffer_class', EnvReplayBuffer)(
        **main_replay_buffer_kwargs,
    )
    if variant.get('use_validation_buffer', False):
        train_replay_buffer = replay_buffer
        validation_replay_buffer = variant.get('replay_buffer_class', EnvReplayBuffer)(
            **main_replay_buffer_kwargs,
        )
        replay_buffer = SplitReplayBuffer(train_replay_buffer, validation_replay_buffer, 0.9)

    trainer_class = variant.get("trainer_class", AWACTrainer)

    trainer = trainer_class(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        buffer_policy=buffer_policy,
        rnd_net1=rnd_net1,
        rnd_net2=rnd_net2,
        goal_model=goal_model,
        **variant['trainer_kwargs']
    )
    if variant['collection_mode'] == 'online':
        expl_path_collector = MdpStepCollector(
            expl_env,
            policy,
        )
        algorithm = TorchOnlineRLAlgorithm(
            trainer=trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            max_path_length=variant['max_path_length'],
            batch_size=variant['batch_size'],
            num_epochs=variant['num_epochs'],
            num_eval_steps_per_epoch=variant['num_eval_steps_per_epoch'],
            num_expl_steps_per_train_loop=variant['num_expl_steps_per_train_loop'],
            num_trains_per_train_loop=variant['num_trains_per_train_loop'],
            min_num_steps_before_training=variant['min_num_steps_before_training'],
        )
    else:
        expl_path_collector = MdpPathCollector(
            expl_env,
            expl_policy,
            name='expl',
            render=variant.get('render_expl', False)
        )
        algorithm = TorchBatchRLAlgorithm(
            trainer=trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            max_path_length=variant['max_path_length'],
            batch_size=variant['batch_size'],
            num_epochs=variant['num_epochs'],
            num_eval_steps_per_epoch=variant['num_eval_steps_per_epoch'],
            num_expl_steps_per_train_loop=variant['num_expl_steps_per_train_loop'],
            num_trains_per_train_loop=variant['num_trains_per_train_loop'],
            min_num_steps_before_training=variant['min_num_steps_before_training'],
        )
    algorithm.to(ptu.device)
    if variant.get('goal_select', False):
        expl_path_collector.goal_select = True
    if variant.get('random_goal', False):
        expl_path_collector.random_goal = True
    if variant.get('use_densities', False):
        expl_path_collector.use_densities = True
    expl_path_collector._replay_buffer=replay_buffer
    expl_path_collector._algo = algorithm

    demo_train_buffer = EnvReplayBuffer(
        **replay_buffer_kwargs,
    )
    demo_test_buffer = EnvReplayBuffer(
        **replay_buffer_kwargs,
    )

    if variant.get("save_video", False):
        if variant.get("presampled_goals", None):
            variant['image_env_kwargs']['presampled_goals'] = load_local_or_remote_file(variant['presampled_goals']).item()

        def get_img_env(env):
            renderer = EnvRenderer(**variant["renderer_kwargs"])
            img_env = InsertImageEnv(GymToMultiEnv(env), renderer=renderer)

        image_eval_env = ImageEnv(GymToMultiEnv(eval_env), **variant["image_env_kwargs"])
        # image_eval_env = get_img_env(eval_env)
        image_eval_path_collector = ObsDictPathCollector(
            image_eval_env,
            eval_policy,
            observation_key="state_observation",
        )
        image_expl_env = ImageEnv(GymToMultiEnv(expl_env), **variant["image_env_kwargs"])
        # image_expl_env = get_img_env(expl_env)
        image_expl_path_collector = ObsDictPathCollector(
            image_expl_env,
            expl_policy,
            observation_key="state_observation",
        )
        video_func = VideoSaveFunction(
            image_eval_env,
            variant,
            image_expl_path_collector,
            image_eval_path_collector,
        )
        algorithm.post_train_funcs.append(video_func)
    if variant.get('save_paths', False):
        algorithm.post_train_funcs.append(save_paths)
    if variant.get('load_demos', False):
        path_loader_class = variant.get('path_loader_class', MDPPathLoader)
        path_loader = path_loader_class(trainer,
            replay_buffer=replay_buffer,
            demo_train_buffer=demo_train_buffer,
            demo_test_buffer=demo_test_buffer,
            **path_loader_kwargs
        )
        path_loader.load_demos()
    if variant.get('load_env_dataset_demos', False):
        path_loader_class = variant.get('path_loader_class', HDF5PathLoader)
        path_loader = path_loader_class(trainer,
            replay_buffer=replay_buffer,
            demo_train_buffer=demo_train_buffer,
            demo_test_buffer=demo_test_buffer,
            **path_loader_kwargs
        )
        path_loader.load_demos(expl_env.get_dataset())
    if variant.get('save_initial_buffers', False):
        buffers = dict(
            replay_buffer=replay_buffer,
            demo_train_buffer=demo_train_buffer,
            demo_test_buffer=demo_test_buffer,
        )
        buffer_path = osp.join(logger.get_snapshot_dir(), 'buffers.p')
        pickle.dump(buffers, open(buffer_path, "wb"))

    if variant.get('pretrained_rl_path', False):
        data = torch.load(variant.get('pretrained_rl_path', False))

        state_dict = data['trainer/policy'].state_dict()
        algorithm.trainer.policy.load_state_dict(state_dict)
        state_dict = data['trainer/qf1'].state_dict()
        algorithm.trainer.qf1.load_state_dict(state_dict)
        state_dict = data['trainer/qf2'].state_dict()
        algorithm.trainer.qf2.load_state_dict(state_dict)
        state_dict = data['trainer/target_qf1'].state_dict()
        algorithm.trainer.target_qf1.load_state_dict(state_dict)
        state_dict = data['trainer/target_qf2'].state_dict()
        algorithm.trainer.target_qf2.load_state_dict(state_dict)
        state_dict = data['trainer/buffer_policy'].state_dict()
        algorithm.trainer.buffer_policy.load_state_dict(state_dict)
        state_dict = data['exploration/policy'].state_dict()
        algorithm.expl_data_collector._policy.load_state_dict(state_dict)
        state_dict = data['evaluation/policy'].state_dict()
        algorithm.eval_data_collector._policy.load_state_dict(state_dict)
        state_dict = data['trainer/goal_model'].state_dict()
        algorithm.trainer.goal_model.load_state_dict(state_dict)

        # Loading optimizers
        algorithm.trainer.optimizers[algorithm.trainer.policy].load_state_dict(
            data['trainer/optimizers'][data['trainer/policy']].state_dict())

        algorithm.trainer.optimizers[algorithm.trainer.goal_model].load_state_dict(
            data['trainer/optimizers'][data['trainer/goal_model']].state_dict())
        print("LOADED IN PRETRAINED PATH")

    if variant.get('pretrain_buffer_policy', False):
        trainer.pretrain_policy_with_bc(
            buffer_policy,
            replay_buffer.train_replay_buffer,
            replay_buffer.validation_replay_buffer,
            10000,
            label="buffer",
        )
    if variant.get('pretrain_policy', False):
        trainer.pretrain_policy_with_bc(
            policy,
            demo_train_buffer,
            demo_test_buffer,
            trainer.bc_num_pretrain_steps,
        )

    if variant.get('pretrain_goal_model', False):
        trainer.pretrain_goalproposer_with_bc(
            goal_model,
            demo_train_buffer,
            demo_test_buffer,
            trainer.bc_goal_num_pretrain_steps,
        )
    if variant.get('pretrain_rl', False):
        trainer.pretrain_q_with_bc_data()

    std_bump = variant.get('std_bump', 0.)
    state_dict = algorithm.trainer.policy.state_dict()
    state_dict['log_std_logits'] += std_bump
    algorithm.trainer.policy.load_state_dict(state_dict)

    if variant.get('save_pretrained_algorithm', False):
        p_path = osp.join(logger.get_snapshot_dir(), 'pretrain_algorithm.p')
        pt_path = osp.join(logger.get_snapshot_dir(), 'pretrain_algorithm.pt')
        data = algorithm._get_snapshot()
        data['algorithm'] = algorithm
        del data['algorithm']
        del data['exploration/env']
        del data['evaluation/env']
        torch.save(data, open(pt_path, "wb"))
        torch.save(data, open(p_path, "wb"))


    # Code for post-hoc evaluation
    if variant.get('post_eval', False):
        load_directory = variant.get('load_directory', None)
        import os
        from pathlib import Path

        files = sorted(Path(load_directory).iterdir(), key=os.path.getmtime)
        diagnostics = []

        num_eval_paths = variant.get('num_eval_paths', 1)
        eval_env.wrapped_env._eval_mode = True

        from rlkit.samplers.rollout_functions import rollout
        import numpy as np
        curr_idx_f = 0
        eval_skip = variant.get('eval_skip', 10)
        starting_idx = variant.get('starting_idx', 0)

        for f in files:
            if ('pkl' not in f.name) or ('itr' not in f.name) or ('diagnostics' in f.name):
                continue

            if curr_idx_f % eval_skip != 0:
                curr_idx_f += 1
                continue

            print("LOADING %s"%(f.name))
            # Load directory
            data = torch.load(f)
            state_dict = data['trainer/policy'].state_dict()
            algorithm.trainer.policy.load_state_dict(state_dict)
            state_dict = data['trainer/qf1'].state_dict()
            algorithm.trainer.qf1.load_state_dict(state_dict)
            state_dict = data['trainer/qf2'].state_dict()
            algorithm.trainer.qf2.load_state_dict(state_dict)
            state_dict = data['trainer/target_qf1'].state_dict()
            algorithm.trainer.target_qf1.load_state_dict(state_dict)
            state_dict = data['trainer/target_qf2'].state_dict()
            algorithm.trainer.target_qf2.load_state_dict(state_dict)
            state_dict = data['trainer/buffer_policy'].state_dict()
            algorithm.trainer.buffer_policy.load_state_dict(state_dict)
            state_dict = data['exploration/policy'].state_dict()
            algorithm.expl_data_collector._policy.load_state_dict(state_dict)
            state_dict = data['evaluation/policy'].state_dict()
            algorithm.eval_data_collector._policy.load_state_dict(state_dict)

            # Loading optimizers
            algorithm.trainer.optimizers[algorithm.trainer.policy].load_state_dict(
                data['trainer/optimizers'][data['trainer/policy']].state_dict())

            algorithm.trainer.optimizers[algorithm.trainer.goal_model].load_state_dict(
                data['trainer/optimizers'][data['trainer/goal_model']].state_dict())

            paths = [[[] for _ in range(eval_env.wrapped_env.goal_matrix.shape[0])] for _ in range(eval_env.wrapped_env.goal_matrix.shape[0])]
            for start_idx in range(eval_env.wrapped_env.goal_matrix.shape[0]):
                eval_env.wrapped_env.commanded_start = start_idx
                viable_goals = np.where(eval_env.wrapped_env.goal_matrix[start_idx] > 0)[0]
                for end_idx in viable_goals:
                    eval_env.wrapped_env.commanded_goal = end_idx
                    for _ in range(num_eval_paths):
                        p = rollout(eval_env, algorithm.trainer.policy, max_path_length=variant.get("max_path_length", 200))
                        paths[start_idx][end_idx].append(p)
            diagnostics.append(paths)
            new_path = os.path.join(load_directory, 'stochastic_diagnostics_itr_%d.pkl'%(curr_idx_f + starting_idx))
            pickle.dump(paths, open(new_path, 'wb'))
            curr_idx_f += 1

        new_path = os.path.join(load_directory, 'stochastic_diagnostics_overall.pkl')
        pickle.dump(diagnostics, open(new_path, 'wb'))


    # Code for post-hoc evaluation
    if variant.get('post_eval_reachability', False):
        load_directory = variant.get('load_directory', None)
        import os
        from pathlib import Path

        files = sorted(Path(load_directory).iterdir(), key=os.path.getmtime)
        diagnostics = []

        import numpy as np
        curr_idx_f = 0
        eval_skip = variant.get('eval_skip', 10)
        # import IPython
        # IPython.embed()
        for f in files:
            if ('pkl' not in f.name) or ('itr' not in f.name) or ('diagnostics' in f.name):
                continue

            if curr_idx_f % eval_skip != 0:
                curr_idx_f += 1
                continue

            print("LOADING %s"%(f.name))
            # Load directory
            data = torch.load(f)
            state_dict = data['trainer/policy'].state_dict()
            algorithm.trainer.policy.load_state_dict(state_dict)
            state_dict = data['trainer/qf1'].state_dict()
            algorithm.trainer.qf1.load_state_dict(state_dict)
            state_dict = data['trainer/qf2'].state_dict()
            algorithm.trainer.qf2.load_state_dict(state_dict)
            state_dict = data['trainer/target_qf1'].state_dict()
            algorithm.trainer.target_qf1.load_state_dict(state_dict)
            state_dict = data['trainer/target_qf2'].state_dict()
            algorithm.trainer.target_qf2.load_state_dict(state_dict)
            state_dict = data['trainer/buffer_policy'].state_dict()
            algorithm.trainer.buffer_policy.load_state_dict(state_dict)
            state_dict = data['exploration/policy'].state_dict()
            algorithm.expl_data_collector._policy.load_state_dict(state_dict)
            state_dict = data['evaluation/policy'].state_dict()
            algorithm.eval_data_collector._policy.load_state_dict(state_dict)

            # Loading optimizers
            algorithm.trainer.optimizers[algorithm.trainer.policy].load_state_dict(
                data['trainer/optimizers'][data['trainer/policy']].state_dict())

            algorithm.trainer.optimizers[algorithm.trainer.goal_model].load_state_dict(
                data['trainer/optimizers'][data['trainer/goal_model']].state_dict())

            stats = algorithm.eval_policy_goalreaching()
            diagnostics.append(stats)
            
            curr_idx_f += 1

            # new_path = os.path.join(load_directory, 'eval_goalreaching.pkl')
            if variant.get('cloud_launch', False):
                new_path = os.path.join(load_directory, 'eval_goalreaching_CORLPAPER.pkl')
            else:
                new_path = os.path.join(logger.get_snapshot_dir(), 'eval_goalreaching_CORLPAPER.pkl')
                
            pickle.dump(diagnostics, open(new_path, 'wb'))
            logger.save_itr_params(0, {'paths': diagnostics})

    if variant.get('train_rl', True):
        algorithm.train()
