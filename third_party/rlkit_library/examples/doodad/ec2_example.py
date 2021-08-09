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

"""
Example of running stuff on EC2
"""
import time

from rlkit.core import logger
from rlkit.launchers.launcher_util import run_experiment
from datetime import datetime
from pytz import timezone
import pytz


def example(variant):
    import torch
    logger.log(torch.__version__)
    date_format = '%m/%d/%Y %H:%M:%S %Z'
    date = datetime.now(tz=pytz.utc)
    logger.log("start")
    logger.log('Current date & time is: {}'.format(date.strftime(date_format)))
    if torch.cuda.is_available():
        x = torch.randn(3)
        logger.log(str(x.to(ptu.device)))

    date = date.astimezone(timezone('US/Pacific'))
    logger.log('Local date & time is: {}'.format(date.strftime(date_format)))
    for i in range(variant['num_seconds']):
        logger.log("Tick, {}".format(i))
        time.sleep(1)
    logger.log("end")
    logger.log('Local date & time is: {}'.format(date.strftime(date_format)))

    logger.log("start mujoco")
    from gym.envs.mujoco import HalfCheetahEnv
    e = HalfCheetahEnv()
    img = e.sim.render(32, 32)
    logger.log(str(sum(img)))
    logger.log("end mujocoy")


if __name__ == "__main__":
    # noinspection PyTypeChecker
    date_format = '%m/%d/%Y %H:%M:%S %Z'
    date = datetime.now(tz=pytz.utc)
    logger.log("start")
    variant = dict(
        num_seconds=10,
        launch_time=str(date.strftime(date_format)),
    )
    run_experiment(
        example,
        exp_prefix="ec2-test",
        mode='ec2',
        variant=variant,
        # use_gpu=True,  # GPUs are much more expensive!
    )
