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

"""Method to handle getting command line arguments.

Don't add additional dependencies here as ths is used by both job_script_mjrl.py
and train.py.
"""

import argparse

def get_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            # Show default value in the help doc.
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument('-c', '--config', nargs='*', default=['job_data_mjrl_0.txt'], help=(
        'Path to the job data config file(s). Multiple config files can be '
        'passed (space-separated list of paths). Globs (*) are supported. '
        'If a directory is given, any .txt file is read as a config.'
    ))
    parser.add_argument('-o', '--output_dir', default='.', help=(
        'Directory to output trained policies, logs, and plots. A subdirectory '
        'is created for each job.'
    ))
    parser.add_argument('-p', '--parallel', action='store_true', help=(
        'Whether to run the jobs in parallel.'
    ))
    parser.add_argument('--num_cpu', default=0, type=int, help=(
        'The number of CPUs to use in each job (for sampling rollouts in '
        'parallel). If not given, the number of CPUs to use is read from the '
        'job config.'
    ))
    return parser.parse_args()
