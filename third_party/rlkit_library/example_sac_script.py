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

"""Script for launching training jobs."""
import copy
import glob
import itertools
import os
os.environ['MUJOCO_GL'] = 'osmesa'
import pickle
import time as timer
import traceback
import pprint

os.environ['MKL_THREADING_LAYER'] = 'GNU'

# Utilities
import train_args
import adept_envs
# parallel job execution
import multiprocessing as mp
import config_reader

from rlkit.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
from rlkit.launchers.experiments.awac.awac_rl import experiment, process_args
from rlkit.launchers.experiments.awac.sac_rl import experiment
import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment

from rlkit.torch.sac.policies import GaussianPolicy
import adept_envs
from rlkit.torch.networks import Clamp
import gym_point
import gym

def single_process(job):
    # Allow process to parallelize things internally
    curr_proc = mp.current_process()
    curr_proc.daemon = False

    # Create a directory for the job results.
    job_dir = os.path.join(job['output_dir'])
    if not os.path.isdir(job_dir):
        os.mkdir(job_dir)

    # start job
    job_start_time = timer.time()
    print('Started New Job : ', job['job_name'], '=======================')
    print('Job specifications : \n', job)

    # job['path_loader_kwargs']['demo_paths'] = list(job['path_loader_kwargs']['demo_paths'])

    # if job['path_loader_class'] == 'DictToMDPPathLoader':
    #     job['path_loader_class'] = DictToMDPPathLoader

    # if job['policy_class'] == 'GaussianPolicy':
    #     job['policy_class'] = GaussianPolicy

    sweeper = hyp.DeterministicHyperparameterSweeper(
        {}, default_parameters=job,
    )

    n_seeds = 1
    mode = 'here_no_doodad'
    exp_prefix = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )
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
                base_log_dir=job['output_dir'],
                gcp_kwargs=dict(
                    zone='us-west1-b',
                ),
            )

    total_job_time = timer.time() - job_start_time
    print('Job', job['job_name'],
          'took %f seconds ==============' % total_job_time)
    return total_job_time


def main():

    # See train_args.py for the list of args.
    args = train_args.get_args()
    # Get the config files, expanding globs and directories (*) if necessary.
    jobs = config_reader.process_config_files(args.config)
    assert jobs, 'No jobs found from config.'

    # Create the output directory if not present.
    output_dir = args.output_dir or os.getcwd()
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    output_dir = os.path.abspath(output_dir)

    for index, job in enumerate(jobs):
        # Modify the job name to include the job number.
        assert 'job_name' in job
        if len(jobs) > 1:
            job['job_name'] = '{}_{}'.format(job['job_name'], index)

        # Add additional parameters to the job.
        job['output_dir'] = os.path.join(output_dir, job['job_name'])

        # Override num_cpus if the args.num_cpu is given or if we're running on hardware.
        job['num_cpu'] = 1

    print('Running {} jobs {}'.format(
        len(jobs), 'in parallel' if args.parallel else 'sequentially'))

    # execute jobs
    t1 = timer.time()
    if args.parallel:
        # processes: Number of processes to create
        # maxtasksperchild: the number of tasks a worker process can complete before it will exit and be replaced with a fresh worker process
        pool = mp.Pool(processes=len(jobs), maxtasksperchild=1)
        parallel_runs = [
            pool.apply_async(single_process, args=(job, )) for job in jobs
        ]
        try:
            max_process_time = 36000  # process time out in seconds
            results = [p.get(timeout=max_process_time) for p in parallel_runs]
        except Exception as e:
            print('exception thrown')
            print(str(e))
            traceback.print_exc()

        pool.close()
        pool.terminate()
        pool.join()
    else:
        for job in jobs:
            try:
                time_taken = single_process(job)
            except Exception as e:
                print('exception thrown')
                print(str(e))
                traceback.print_exc()

    t2 = timer.time()
    print('Total time taken = ', t2 - t1)
    return


if __name__ == '__main__':
    main()
