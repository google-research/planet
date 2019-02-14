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
from tensorflow_probability import distributions as tfd


class MSEDistribution(object):

  def __init__(self, mean):
    """Gaussian with negative squared error as log probability.

    The log_prob() method computes the sum of the element-wise squared
    distances. This means that its value is both unnormalized and does not
    depend on the standard deviation.

    Args:
      mean: Mean of the distribution.
      stddev: Standard deviation, ignored by log_prob().
    """
    self._dist = tfd.Normal(mean, 1.0)
    self._mean = mean

  def __getattr__(self, name):
    return getattr(self._dist, name)

  def log_prob(self, event):
    squared_error = -((event - self._mean) ** 2)
    if self.event_shape.ndims:
      event_dims = [-(x + 1) for x in range(self.event_shape).ndims]
      squared_error = tf.reduce_sum(squared_error, event_dims)
    return squared_error
