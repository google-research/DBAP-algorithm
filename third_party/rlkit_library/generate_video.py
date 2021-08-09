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

from PIL import Image
import glob
import os
import numpy as np
import skvideo.io
image_list_expl = []
image_list_eval = []
import glob
import os
# path = "/usr/local/google/home/abhishekunique/sim_franka/rlkit/20201230-07h36m-lowervariance-targeteddata-backforth/test-4/test-4/12-30-dev-example-awac-script/12-30-dev-example-awac-script_2020_12_30_15_38_43_0000--s-76665/*.png"
path = "/usr/local/google/home/abhishekunique/sim_franka/rlkit/20201230-07h36m-lowervariance-targeteddata-backforth/test-36/test-36/12-30-dev-example-awac-script/12-30-dev-example-awac-script_2020_12_30_15_38_42_0000--s-7314/*.png"
files = glob.glob(path)
files.sort(key=os.path.getmtime)
for filename in files: #assuming gif
    im=Image.open(filename)
    if 'expl' in filename:
        image_list_expl.append(np.array(im))
    elif 'eval' in filename:
        image_list_eval.append(np.array(im))
skvideo.io.vwrite('eval_vid.mp4', np.array(image_list_eval))
skvideo.io.vwrite('expl_vid.mp4', np.array(image_list_expl))

