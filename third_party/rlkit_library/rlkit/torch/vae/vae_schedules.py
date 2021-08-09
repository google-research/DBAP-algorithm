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

def always_train(epoch):
    return True, 300


def custom_schedule(epoch):
    if epoch < 10:
        return True, 1000
    elif epoch < 300:
        return True, 200
    else:
        return epoch % 3 == 0, 200


def custom_schedule_2(epoch):
    if epoch < 10:
        return True, 1000
    elif epoch < 100:
        return True, 200
    else:
        return epoch % 2 == 0, 200


def every_other(epoch):
    return epoch % 2 == 0, 400


def every_three(epoch):
    return epoch % 3 == 0, 600


def every_three_a_lot(epoch):
    return epoch % 3 == 0, 1200


def every_six(epoch):
    return epoch % 6 == 0, 1200


def every_six_less(epoch):
    return epoch % 6 == 0, 600


def every_six_much_less(epoch):
    return epoch % 6 == 0, 300


def every_ten(epoch):
    return epoch % 10 == 0 or epoch == 5, 1000


def every_twenty(epoch):
    return epoch % 10 == 0 or epoch == 5 or epoch == 10, 1000


def never_train(epoch):
    return False, 0
