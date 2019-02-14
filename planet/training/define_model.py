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

import tensorflow as tf

from planet import tools
from planet.training import define_summaries
from planet.training import utility


def define_model(data, trainer, config):
  tf.logging.info('Build TensorFlow compute graph.')
  dependencies = []
  step = trainer.step
  global_step = trainer.global_step
  phase = trainer.phase
  should_summarize = trainer.log

  # Preprocess data.
  with tf.device('/cpu:0'):
    if config.dynamic_action_noise:
      data['action'] += tf.random_normal(
          tf.shape(data['action']), 0.0, config.dynamic_action_noise)
    prev_action = tf.concat(
        [0 * data['action'][:, :1], data['action'][:, :-1]], 1)
    obs = data.copy()
    del obs['length']

  # Instantiate network blocks.
  cell = config.cell()
  kwargs = dict()
  encoder = tf.make_template(
      'encoder', config.encoder, create_scope_now_=True, **kwargs)
  heads = {}
  for key, head in config.heads.items():
    name = 'head_{}'.format(key)
    kwargs = dict(data_shape=obs[key].shape[2:].as_list())
    heads[key] = tf.make_template(name, head, create_scope_now_=True, **kwargs)

  # Embed observations and unroll model.
  embedded = encoder(obs)
  # Separate overshooting and zero step observations because computing
  # overshooting targets for images would be expensive.
  zero_step_obs = {}
  overshooting_obs = {}
  for key, value in obs.items():
    if config.zero_step_losses.get(key):
      zero_step_obs[key] = value
    if config.overshooting_losses.get(key):
      overshooting_obs[key] = value
  assert config.overshooting <= config.batch_shape[1]
  target, prior, posterior, mask = tools.overshooting(
      cell, overshooting_obs, embedded, prev_action, data['length'],
      config.overshooting + 1)
  losses = []

  # Zero step losses.
  _, zs_prior, zs_posterior, zs_mask = tools.nested.map(
      lambda tensor: tensor[:, :, :1], (target, prior, posterior, mask))
  zs_target = {key: value[:, :, None] for key, value in zero_step_obs.items()}
  zero_step_losses = utility.compute_losses(
      config.zero_step_losses, cell, heads, step, zs_target, zs_prior,
      zs_posterior, zs_mask, config.free_nats, debug=config.debug)
  losses += [
      loss * config.zero_step_losses[name] for name, loss in
      zero_step_losses.items()]
  if 'divergence' not in zero_step_losses:
    zero_step_losses['divergence'] = tf.zeros((), dtype=tf.float32)

  # Overshooting losses.
  if config.overshooting > 1:
    os_target, os_prior, os_posterior, os_mask = tools.nested.map(
        lambda tensor: tensor[:, :, 1:-1], (target, prior, posterior, mask))
    if config.stop_os_posterior_gradient:
      os_posterior = tools.nested.map(tf.stop_gradient, os_posterior)
    overshooting_losses = utility.compute_losses(
        config.overshooting_losses, cell, heads, step, os_target, os_prior,
        os_posterior, os_mask, config.free_nats, debug=config.debug)
    losses += [
        loss * config.overshooting_losses[name] for name, loss in
        overshooting_losses.items()]
  else:
    overshooting_losses = {}
  if 'divergence' not in overshooting_losses:
    overshooting_losses['divergence'] = tf.zeros((), dtype=tf.float32)

  # Workaround for TensorFlow deadlock bug.
  loss = sum(losses)
  train_loss = tf.cond(
      tf.equal(phase, 'train'),
      lambda: loss,
      lambda: 0 * tf.get_variable('dummy_loss', (), tf.float32))
  train_summary = utility.apply_optimizers(
      train_loss, step, should_summarize, config.optimizers)
  # train_summary = tf.cond(
  #     tf.equal(phase, 'train'),
  #     lambda: utility.apply_optimizers(
  #         loss, step, should_summarize, config.optimizers),
  #     str, name='optimizers')

  # Active data collection.
  collect_summaries = []
  graph = tools.AttrDict(locals())
  with tf.variable_scope('collection'):
    should_collects = []
    for name, params in config.sim_collects.items():
      after, every = params.steps_after, params.steps_every
      should_collect = tf.logical_and(
          tf.equal(phase, 'train'),
          tools.schedule.binary(step, config.batch_shape[0], after, every))
      collect_summary, _ = tf.cond(
          should_collect,
          functools.partial(
              utility.simulate_episodes, config, params, graph, name),
          lambda: (tf.constant(''), tf.constant(0.0)),
          name='should_collect_' + params.task.name)
      should_collects.append(should_collect)
      collect_summaries.append(collect_summary)

  # Compute summaries.
  graph = tools.AttrDict(locals())
  with tf.control_dependencies(collect_summaries):
    summaries, score = tf.cond(
        should_summarize,
        lambda: define_summaries.define_summaries(graph, config),
        lambda: (tf.constant(''), tf.zeros((0,), tf.float32)),
        name='summaries')
  with tf.device('/cpu:0'):
    summaries = tf.summary.merge([summaries, train_summary])
    # summaries = tf.summary.merge([summaries, train_summary] + collect_summaries)
    zs_entropy = (tf.reduce_sum(tools.mask(
        cell.dist_from_state(zs_posterior, zs_mask).entropy(), zs_mask)) /
        tf.reduce_sum(tf.to_float(zs_mask)))
    dependencies.append(utility.print_metrics((
        ('score', score),
        ('loss', loss),
        ('zs_entropy', zs_entropy),
        ('zs_divergence', zero_step_losses['divergence']),
    ), step, config.mean_metrics_every))
  with tf.control_dependencies(dependencies):
    score = tf.identity(score)
  return score, summaries
