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

import functools
import logging
import os

import ruamel.yaml as yaml
import tensorflow as tf

from planet import control
from planet import tools
from planet.training import trainer as trainer_


def set_up_logging():
  """Configure the TensorFlow logger."""
  tf.logging.set_verbosity(tf.logging.INFO)
  logging.getLogger('tensorflow').propagate = False
  logging.getLogger('tensorflow').format = '%(message)s'
  logging.basicConfig(level=logging.INFO, format='%(message)s')


def save_config(config, logdir=None):
  """Save a new configuration by name.

  If a logging directory is specified, is will be created and the configuration
  will be stored there. Otherwise, a log message will be printed.

  Args:
    config: Configuration object.
    logdir: Location for writing summaries and checkpoints if specified.

  Returns:
    Configuration object.
  """
  if logdir:
    with config.unlocked:
      config.logdir = logdir
    message = 'Start a new run and write summaries and checkpoints to {}.'
    tf.logging.info(message.format(config.logdir))
    tf.gfile.MakeDirs(config.logdir)
    config_path = os.path.join(config.logdir, 'config.yaml')
    with tf.gfile.GFile(config_path, 'w') as file_:
      yaml.dump(
          config, file_, yaml.Dumper,
          allow_unicode=True,
          default_flow_style=False)
  else:
    message = (
        'Start a new run without storing summaries and checkpoints since no '
        'logging directory was specified.')
    tf.logging.info(message)
  return config


def load_config(logdir):
  """Load a configuration from the log directory.

  Args:
    logdir: The logging directory containing the configuration file.

  Raises:
    IOError: The logging directory does not contain a configuration file.

  Returns:
    Configuration object.
  """
  config_path = logdir and os.path.join(logdir, 'config.yaml')
  if not config_path or not tf.gfile.Exists(config_path):
    message = (
        'Cannot resume an existing run since the logging directory does not '
        'contain a configuration file.')
    raise IOError(message)
  try:
    with tf.gfile.GFile(config_path, 'r') as file_:
      config = yaml.load(file_, yaml.Loader)
      message = 'Resume run and write summaries and checkpoints to {}.'
      tf.logging.info(message.format(config.logdir))
  except Exception:
    raise IOError('yaml is still broken.')
  return config


def get_batch(datasets, phase, reset):
  """Read batches from multiple datasets based on the training phase.

  The test dataset is reset at the beginning of every test phase. The training
  dataset is repeated infinitely and doesn't need a reset.

  Args:
    datasets: Dictionary of datasets with training phases as keys.
    phase: Tensor of the training phase name.
    reset: Whether to reset the datasets.

  Returns:
    data: a batch of data from either the train or test set.
  """
  with datasets.unlocked:
    datasets.train = datasets.train.make_one_shot_iterator()
    datasets.test = datasets.test.make_one_shot_iterator()
  data = tf.cond(
      tf.equal(phase, 'train'),
      datasets.train.get_next,
      datasets.test.get_next)
  if not isinstance(data, dict):
    data = {'data': data}
  if 'length' not in data:
    example = data[list(data.keys())[0]]
    data['length'] = (
        tf.zeros((tf.shape(example)[0],), tf.int32) + tf.shape(example)[1])
  return data


def train(model_fn, datasets, logdir, config):
  """Train a model on a datasets.

  The model function receives the following arguments: data batch, trainer
  phase, whether it should log, and the config. The configuration object should
  contain the attributes `batch_shape`, `train_steps`, `test_steps`,
  `max_steps`, in addition to the attributes expected by the model function.

  Args:
    model_fn: Function greating the model graph.
    datasets: Dictionary with keys `train` and `test` and datasets as values.
    logdir: Optional logging directory for summaries and checkpoints.
    config: Configuration object.

  Yields:
    Test score of every epoch.

  Raises:
    KeyError: if config is falsey.
  """
  if not config:
    raise KeyError('You must specify a configuration.')
  logdir = logdir and os.path.expanduser(logdir)
  try:
    config = load_config(logdir)
  except IOError:
    config = save_config(config, logdir)
  trainer = trainer_.Trainer(logdir, config=config)
  with tf.variable_scope('graph', use_resource=True):
    data = get_batch(datasets, trainer.phase, trainer.reset)
    score, summary = model_fn(data, trainer, config)
    message = 'Graph contains {} trainable variables.'
    tf.logging.info(message.format(tools.count_weights()))
    if config.train_steps:
      trainer.add_phase(
          'train', config.train_steps, score, summary,
          batch_size=config.batch_shape[0],
          report_every=None,
          log_every=config.train_log_every,
          checkpoint_every=config.train_checkpoint_every)
    if config.test_steps:
      trainer.add_phase(
          'test', config.test_steps, score, summary,
          batch_size=config.batch_shape[0],
          report_every=config.test_steps,
          log_every=config.test_steps,
          checkpoint_every=config.test_checkpoint_every)
  for saver in config.savers:
    trainer.add_saver(**saver)
  for score in trainer.iterate(config.max_steps):
    yield score


def compute_losses(
    loss_scales, cell, heads, step, target, prior, posterior, mask,
    free_nats=None, debug=False):
  features = cell.features_from_state(posterior)
  losses = {}
  for key, scale in loss_scales.items():
    # Skip losses with zero or None scale to save computation.
    if not scale:
      continue
    elif key == 'divergence':
      loss = cell.divergence_from_states(posterior, prior, mask)
      if free_nats is not None:
        loss = tf.maximum(tf.cast(free_nats, tf.float32), loss)
      loss = tf.reduce_sum(loss, 1) / tf.reduce_sum(tf.to_float(mask), 1)
    elif key == 'global_divergence':
      global_prior = {
          'mean': tf.zeros_like(prior['mean']),
          'stddev': tf.ones_like(prior['stddev'])}
      loss = cell.divergence_from_states(posterior, global_prior, mask)
      loss = tf.reduce_sum(loss, 1) / tf.reduce_sum(tf.to_float(mask), 1)
    elif key in heads:
      output = heads[key](features)
      loss = -tools.mask(output.log_prob(target[key]), mask)
    else:
      message = "Loss scale references unknown head '{}'."
      raise KeyError(message.format(key))
    # Average over the batch and normalize by the maximum chunk length.
    loss = tf.reduce_mean(loss)
    losses[key] = tf.check_numerics(loss, key) if debug else loss
  return losses


def apply_optimizers(loss, step, should_summarize, optimizers):
  summaries = []
  training_ops = []
  for name, optimizer_cls in optimizers.items():
    with tf.variable_scope('optimizer_{}'.format(name)):
      optimizer = optimizer_cls(step=step, should_summarize=should_summarize)
      optimize, opt_summary = optimizer.minimize(loss)
      training_ops.append(optimize)
      summaries.append(opt_summary)
  with tf.control_dependencies(training_ops):
    return tf.cond(should_summarize, lambda: tf.summary.merge(summaries), str)


def simulate_episodes(config, params, graph, name):
  def env_ctor():
    env = params.task.env_ctor()
    if params.save_episode_dir:
      env = control.wrappers.CollectGymDataset(env, params.save_episode_dir)
    env = control.wrappers.ConcatObservation(env, ['image'])
    return env
  cell = graph.cell
  agent_config = tools.AttrDict(
      cell=cell,
      encoder=graph.encoder,
      planner=params.planner,
      objective=functools.partial(params.objective, graph=graph),
      exploration=params.exploration,
      preprocess_fn=config.preprocess_fn,
      postprocess_fn=config.postprocess_fn)
  params = params.copy()
  params.update(agent_config)
  agent_config.update(params)
  # Batch size larger crashes so we simulate the episodes individually.
  summaries, returns = [], []
  for index in range(params.batch_size):
    # with tf.control_dependencies(summaries + returns):
    with tf.variable_scope('simulate-{}'.format(index + 1)):
      summary, return_ = control.simulate(
          graph.step, env_ctor, params.task.max_length,
          1, agent_config, name=name)
    summaries.append(summary)
    returns.append(return_)
  summary = tf.summary.merge(summaries)
  return_ = tf.reduce_mean(returns)
  return summary, return_


def print_metrics(metrics, step, every):
  means, updates = [], []
  for key, value in metrics:
    mean = tools.StreamingMean((), tf.float32, 'mean_{}'.format(key))
    means.append(mean)
    updates.append(mean.submit(value))
  with tf.control_dependencies(updates):
    message = 'step/' + '/'.join(key for key, _ in metrics) + ' = '
    gs = tf.train.get_or_create_global_step()
    print_metrics = tf.cond(
        tf.equal(step % every, 0),
        lambda: tf.print(message, [gs] + [mean.clear() for mean in means]),
        tf.no_op)
  return print_metrics


def collect_initial_episodes(config):
  if config.source_train_dir:
    tools.copy_directory(
        config.source_train_dir, config.train_dir,
        config.source_train_amount)
  if config.source_test_dir:
    tools.copy_directory(
        config.source_test_dir, config.test_dir,
        config.source_test_amount)
  items = config.random_collects.items()
  items = sorted(items, key=lambda x: x[0])
  for name, params in items:
    message = 'Collecting {}+ random episodes ({}).'
    tf.logging.info(message.format(params.num_episodes, name))
    control.random_episodes(
        params.task.env_ctor,
        params.num_episodes,
        params.save_episode_dir)
