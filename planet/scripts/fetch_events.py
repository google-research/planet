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

import argparse
import csv
import errno
import functools
import multiprocessing.dummy as multiprocessing
import os
import re
import sys
import traceback

# import imageio
import numpy as np
import skimage.io
import tensorflow as tf
from tensorboard.backend.event_processing import (
    plugin_event_multiplexer as event_multiplexer)


DEFAULT_TAGS = {
    'score': 'trainer/graph/phase_test/cond_2/trainer/test/score',
    'rollout': 'graph/summaries/simulation/cond/cem-reward-12/image/image/0',
}


def create_reader(logdir):
  reader = event_multiplexer.EventMultiplexer(
      tensor_size_guidance={'scalars': 1000})
  reader.AddRunsFromDirectory(logdir)
  reader.Reload()
  return reader


def extract_values(reader, run, tag):
  events = reader.Tensors(run, tag)
  steps = [event.step for event in events]
  times = [event.wall_time for event in events]
  values = [tf.make_ndarray(event.tensor_proto) for event in events]
  return steps, times, values


def export_scalar(filepath, steps, times, values):
  values = [value.item() for value in values]
  with open(filepath + '.csv', 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(('wall_time', 'step', 'value'))
    for row in zip(times, steps, values):
      writer.writerow(row)


def export_image(filepath, steps, times, values):
  tf.reset_default_graph()
  tf_string = tf.placeholder(tf.string)
  tf_tensor = tf.image.decode_image(tf_string)
  with tf.Session() as sess:
    for step, time, value in zip(steps, times, values):
      filename = '{}-{}-{}.png'.format(filepath, step, time)
      width, height, string = value[0], value[1], value[2]
      tensor = sess.run(tf_tensor, {tf_string: string})
      # imageio.imsave(filename, tensor)
      skimage.io.imsave(filename, tensor)
      filename = '{}-{}-{}.npy'.format(filepath, step, time)
      np.save(filename, tensor)


def process_logdir(logdir, lock, args):
  reader = create_reader(logdir)
  runs = tf.gfile.Glob(os.path.join(logdir, args.runs))
  for run in runs:
    filename = re.sub('[^A-Za-z0-9_]', '_', '{}___{}'.format(run, args.tag))
    filepath = os.path.join(args.outdir, filename)
    if os.path.exists(filepath) and not args.force:
      with lock:
        print('Exists', run)
      return
    try:
      with lock:
        print('Start', run)
        steps, times, values = extract_values(
            reader, os.path.relpath(run, logdir), args.tag)
        if args.type == 'scalar':
          export_scalar(filepath, steps, times, values)
        elif args.type == 'image':
          export_image(filepath, steps, times, values)
        else:
          message = "Unsupported summary type '{}'."
          raise NotImplementedError(message.format(args.type))
      with lock:
        print('Done', run)
    except Exception:
      with lock:
        print('Exception', run)
        print(traceback.print_exc())


def main(args):
  logdirs = tf.gfile.Glob(args.logdirs)
  print(len(logdirs), 'logdirs.')
  assert logdirs
  tf.gfile.MakeDirs(args.outdir)
  pool = multiprocessing.Pool(args.workers)
  lock = multiprocessing.Lock()
  pool.map(functools.partial(process_logdir, lock=lock, args=args), logdirs)


if __name__ == '__main__':
  boolean = lambda x: ['False', 'True'].index(x)
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--logdirs', required=True,
      help='glob for directories to scan')
  parser.add_argument(
      '--runs', default='test',
      help='glob for sub-directories to read from')
  parser.add_argument(
      '--tag', default='score',
      help='summary name to read')
  parser.add_argument(
      '--type', default='scalar', choices=['scalar', 'image'],
      help='summary type')
  parser.add_argument(
      '--outdir', required=True,
      help='output directory to store CSV files')
  parser.add_argument(
      '--force', type=boolean, default=False,
      help='overwrite existing files')
  parser.add_argument(
      '--workers', type=int, default=10,
      help='number of worker threads')
  args_, remaining = parser.parse_known_args()
  args_.logdirs = os.path.expanduser(args_.logdirs)
  args_.outdir = os.path.expanduser(args_.outdir)
  if args_.tag in DEFAULT_TAGS:
    args_.tag = DEFAULT_TAGS[args_.tag]
  remaining.insert(0, sys.argv[0])
  tf.app.run(lambda _: main(args_), remaining)
