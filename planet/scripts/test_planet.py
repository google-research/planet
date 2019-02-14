# Copyright 2019 The PlaNet Authors. All rights reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from planet import tools
from planet.scripts import train


class PlanetTest(tf.test.TestCase):

  def test_default(self):
    args = tools.AttrDict(
        logdir=self.create_tempdir(),
        num_runs=1,
        config='debug',
        params=tools.AttrDict(
            task='cheetah_run',
            train_steps=10,
            test_steps=10,
            max_steps=50,
            batch_size=(5, 10)),
        ping_every=0,
        resume_runs=False)
    tf.app.run(lambda _: train.main(args), [sys.argv[0]])

  def test_no_overshooting(self):
    args = tools.AttrDict(
        logdir=self.create_tempdir(),
        num_runs=1,
        config='debug',
        params=tools.AttrDict(
            task='cheetah_run',
            train_steps=10,
            test_steps=10,
            max_steps=50,
            batch_size=(5, 10),
            overshooting=0),
        ping_every=0,
        resume_runs=False)
    tf.app.run(lambda _: train.main(args), [sys.argv[0]])
