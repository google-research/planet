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

"""Load tensors from a directory of numpy files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import random

from scipy.ndimage import interpolation
import numpy as np
import tensorflow as tf

from planet.tools import attr_dict
from planet.tools import chunk_sequence


def numpy_episodes(
    train_dir, test_dir, shape, loader, preprocess_fn=None, scan_every=10,
    num_chunks=None, **kwargs):
  """Read sequences stored as compressed Numpy files as a TensorFlow dataset.

  Args:
    train_dir: Directory containing NPZ files of the training dataset.
    test_dir: Directory containing NPZ files of the testing dataset.
    shape: Tuple of batch size and chunk length for the datasets.
    use_cache: Boolean. Set to True to cache episodes in memory. Default is to
        read episodes from disk every time.
    **kwargs: Keyword arguments to forward to the read episodes implementation.

  Returns:
    Structured data from numpy episodes as Tensors.
  """
  try:
    dtypes, shapes = _read_spec(train_dir, **kwargs)
  except ZeroDivisionError:
    dtypes, shapes = _read_spec(test_dir, **kwargs)
  loader = {
      'scan': functools.partial(_read_episodes_scan, every=scan_every),
      'reload': _read_episodes_reload,
      'dummy': _read_episodes_dummy,
  }[loader]
  train = tf.data.Dataset.from_generator(
      functools.partial(loader, train_dir, shape[0], **kwargs), dtypes, shapes)
  test = tf.data.Dataset.from_generator(
      functools.partial(loader, test_dir, shape[0], **kwargs), dtypes, shapes)
  chunking = lambda x: tf.data.Dataset.from_tensor_slices(
      chunk_sequence(x, shape[1], True, num_chunks))
  def sequence_preprocess_fn(sequence):
    if preprocess_fn:
      with tf.device('/cpu:0'):
        sequence['image'] = preprocess_fn(sequence['image'])
    return sequence
  train = train.flat_map(chunking)
  train = train.shuffle(100 * shape[0])
  train = train.batch(shape[0], drop_remainder=True)
  train = train.map(sequence_preprocess_fn, 10).prefetch(20)
  test = test.flat_map(chunking)
  test = test.shuffle(100 * shape[0])
  test = test.batch(shape[0], drop_remainder=True)
  test = test.map(sequence_preprocess_fn, 10).prefetch(20)
  return attr_dict.AttrDict(train=train, test=test)


def _read_spec(directory, return_length=False, numpy_types=False, **kwargs):
  episodes = _read_episodes_reload(directory, batch_size=1, **kwargs)
  episode = next(episodes)
  episodes.close()
  dtypes = {key: value.dtype for key, value in episode.items()}
  if not numpy_types:
    dtypes = {key: tf.as_dtype(value) for key, value in dtypes.items()}
  shapes = {key: value.shape for key, value in episode.items()}
  shapes = {key: (None,) + shape[1:] for key, shape in shapes.items()}
  if return_length:
    length = len(episode[list(shapes.keys())[0]])
    return dtypes, shapes, length
  else:
    return dtypes, shapes


def _read_episodes_scan(
    directory, batch_size, every, max_episodes=None, **kwargs):
  recent = {}
  cache = {}
  while True:
    index = 0
    for episode in np.random.permutation(list(recent.values())):
      yield episode
      index += 1
      if index > every / 2:
        break
    for episode in np.random.permutation(list(cache.values())):
      index += 1
      yield episode
      if index > every:
        break
    # At the end of the epoch, add new episodes to the cache.
    cache.update(recent)
    recent = {}
    filenames = tf.gfile.Glob(os.path.join(directory, '*.npz'))
    filenames = [filename for filename in filenames if filename not in cache]
    if max_episodes:
      filenames = filenames[:max_episodes - len(cache)]
    for filename in filenames:
      recent[filename] = _read_episode(filename, **kwargs)


def _read_episodes_reload(directory, batch_size, max_episodes=None, **kwargs):
  directory = os.path.expanduser(directory)
  while True:
    filenames = tf.gfile.Glob(os.path.join(directory, '*.npz'))
    if len(filenames) < batch_size:
      # Make sure there are at least batch_size files, since one file gives one
      # sample per call of _read_episodes.
      filenames *= ((batch_size // len(filenames)) + 1)
      random.shuffle(filenames)
      filenames = filenames[:batch_size]
    if max_episodes:
      filenames = list(sorted(filenames))[:max_episodes]
    random.shuffle(filenames)  # In place.
    for filename in filenames:
      yield _read_episode(filename, **kwargs)


def _read_episodes_dummy(
    directory, batch_size, max_episodes=None, **kwargs):
  random = np.random.RandomState(seed=0)
  dtypes, shapes, length = _read_spec(directory, True, True, **kwargs)
  while True:
    episode = {}
    for key in dtypes:
      dtype, shape = dtypes[key], (length,) + shapes[key][1:]
      if dtype in (np.float32, np.float64):
        episode[key] = random.uniform(0, 1, shape).astype(dtype)
      elif dtype in (np.int32, np.int64, np.uint8):
        episode[key] = random.uniform(0, 255, shape).astype(dtype)
      else:
        raise NotImplementedError('Unsupported dtype {}.'.format(dtype))
    yield episode


def _read_episode(
    filename, resize=None, sub_sample=None, max_length=None,
    action_noise=None):
  with tf.gfile.Open(filename, 'rb') as file_:
    episode = np.load(file_)
  episode = {key: _convert_type(episode[key]) for key in episode.keys()}
  for key in ('bias', 'discount'):
    if key in episode:
      del episode[key]
  if sub_sample and sub_sample > 1:
    episode = {key: value[::sub_sample] for key, value in episode.items()}
  if max_length:
    episode = {key: value[:max_length] for key, value in episode.items()}
  if resize and resize != 1:
    factors = (1, resize, resize, 1)
    episode['image'] = interpolation.zoom(episode['image'], factors)
  if action_noise:
    seed = np.fromstring(filename, dtype=np.uint8)
    episode['action'] += np.random.RandomState(seed).normal(
        0, action_noise, episode['action'].shape)
  return episode


def _convert_type(array):
  if array.dtype == np.float64:
    return array.astype(np.float32)
  if array.dtype == np.int64:
    return array.astype(np.int32)
  return array
