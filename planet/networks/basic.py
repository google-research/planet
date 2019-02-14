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

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from planet import tools


def feed_forward(
    state, data_shape, num_layers=2, activation=None, cut_gradient=False):
  """Create a model returning unnormalized MSE distribution."""
  hidden = state
  if cut_gradient:
    hidden = tf.stop_gradient(hidden)
  for _ in range(num_layers):
    hidden = tf.layers.dense(hidden, 100, tf.nn.relu)
  mean = tf.layers.dense(hidden, int(np.prod(data_shape)), activation)
  mean = tf.reshape(mean, tools.shape(state)[:-1] + data_shape)
  dist = tools.MSEDistribution(mean)
  dist = tfd.Independent(dist, len(data_shape))
  return dist
