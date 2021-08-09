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

import numpy as np

import sys
# print(sys.path)
sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")

import cv2
import sys
import pickle

def play_demos(path):
    data = pickle.load(open(path, "rb"))
    # data = np.load(path, allow_pickle=True)

    for traj in data:
        obs = traj["observations"]

        for o in obs:
            img = o["image_observation"].reshape(3, 500, 300)[:, 60:, :240].transpose()
            img = img[:, :, ::-1]
            cv2.imshow('window', img)
            cv2.waitKey(100)

if __name__ == '__main__':
    demo_path = sys.argv[1]
    play_demos(demo_path)
