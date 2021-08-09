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

try:
    import ace
except ImportError:
    print("ERROR: ace not found on your PYTHONPATH.")
    raise

import glob
import os

from ace.generators.config_generator import (
    expand_configs,
    load_config_file,
    Ref
)


def process_config_files(config_file_paths, default_ext='.txt'):
    """Finds and processes config files in the given list of paths.

    Args:
        config_file_paths: List of paths. Globs (*) will automatically be
            expanded. Directories will be searched for files of the config
            extension type.
        default_ext: The default extension type of configs. Directory paths will
            be searched for this extension type.
    Returns:
        A list of expanded configurations from the files.
    """
    if isinstance(config_file_paths, str):
        config_file_paths = [config_file_paths]

    paths = []
    # Expand globs and directories.
    for path in config_file_paths:
        if '*' in path:
            paths.extend(glob.glob(path))
        elif os.path.isdir(path):
            paths.extend(glob.glob(os.path.join(path, '*' + default_ext)))
        else:
            paths.append(path)
    # Add jobs from config files.
    configs = []
    for path in sorted(paths):
        path_configs = process_config_file(path)
        configs.extend(path_configs)
        print('Added {} configs from {}'.format(len(path_configs), path))
    return configs


def process_config_file(config_file_path):
    """Returns a list of expanded configurations from the given file path."""
    with open(config_file_path, 'r') as file:
        return process_config(file.read())


def process_config(config_str):
    """Returns a list of expanded configurations from the given string."""
    config_value = eval(config_str)
    return expand_configs(config_value)
