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

"""Copmute discounted return."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def discounted_return(reward, length, discount):
  """Discounted Monte-Carlo returns."""
  timestep = tf.range(reward.shape[1].value)
  mask = tf.cast(timestep[None, :] < length[:, None], tf.float32)
  return_ = tf.reverse(tf.transpose(tf.scan(
      lambda agg, cur: cur + discount * agg,
      tf.transpose(tf.reverse(mask * reward, [1]), [1, 0]),
      tf.zeros_like(reward[:, -1]), 1, False), [1, 0]), [1])
  return tf.stop_gradient(return_)
