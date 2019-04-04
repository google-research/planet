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
import os

import numpy as np
import tensorflow as tf

from planet import control
from planet import models
from planet import networks
from planet import tools
from planet.scripts import tasks as tasks_lib


def default(config, params):
  config.debug = False
  config.zero_step_losses = tools.AttrDict(_unlocked=True)
  config = _data_processing(config, params)
  config = _model_components(config, params)
  config = _tasks(config, params)
  config = _loss_functions(config, params)
  config = _training_schedule(config, params)
  return config


def debug(config, params):
  with params.unlocked:
    params.collect_every = 20
    params.batch_shape = [5, 10]
    params.train_steps = 30
    params.test_steps = 30
    params.max_steps = 100 * (30 * 30)
    params.collect_every = 20
    params.num_seed_episodes = 2
  config = default(config, params)
  config.debug = True
  return config


def _data_processing(config, params):
  config.max_episodes = None
  config.scan_episodes_every = params.get('scan_episodes_every', 10)
  config.data_loader = params.get('data_loader', 'scan')
  config.batch_shape = params.get('batch_shape', (50, 50))
  config.num_chunks = params.get('num_chunks', 1)
  image_bits = params.get('image_bits', 5)
  config.preprocess_fn = functools.partial(
      tools.preprocess.preprocess, bits=image_bits)
  config.postprocess_fn = functools.partial(
      tools.preprocess.postprocess, bits=image_bits)
  config.open_loop_context = 5
  return config


def _model_components(config, params):
  network = getattr(networks, params.get('network', 'conv_ha'))
  config.encoder = network.encoder
  config.decoder = network.decoder
  config.heads = tools.AttrDict(image=config.decoder)
  size = params.get('model_size', 200)
  state_size = params.get('state_size', 30)
  model = params.get('model', 'rssm')
  if model == 'ssm':
    config.cell = functools.partial(
        models.SSM, state_size, size,
        params.get('mean_only', False),
        params.get('min_stddev', 1e-1))
  elif model == 'rssm':
    config.cell = functools.partial(
        models.RSSM, state_size, size, size,
        params.get('future_rnn', False),
        params.get('mean_only', False),
        params.get('min_stddev', 1e-1))
  else:
    raise NotImplementedError("Unknown model '{}.".format(params.model))
  return config


def _tasks(config, params):
  tasks = params.get('tasks', ['cheetah_run'])
  if tasks == 'all':
    tasks = [
        'cartpole_balance', 'cartpole_swingup', 'finger_spin', 'cheetah_run',
        'cup_catch', 'walker_walk']
  tasks = [getattr(tasks_lib, name)(config, params) for name in tasks]
  config.isolate_envs = params.get('isolate_envs', 'thread')
  def common_spaces_ctor(task, action_spaces):
    env = task.env_ctor()
    env = control.wrappers.SelectObservations(env, ['image'])
    env = control.wrappers.PadActions(env, action_spaces)
    return env
  if len(tasks) > 1:
    action_spaces = [task.env_ctor().action_space for task in tasks]
    for index, task in enumerate(tasks):
      env_ctor = functools.partial(common_spaces_ctor, task, action_spaces)
      tasks[index] = tasks_lib.Task(
          task.name, env_ctor, task.max_length, ['reward'])
  for name in tasks[0].state_components:
    config.heads[name] = networks.feed_forward
    config.zero_step_losses[name] = 1.0
  config.tasks = tasks
  return config


def _loss_functions(config, params):
  config.free_nats = params.get('free_nats', 2.0)
  config.stop_os_posterior_gradient = True
  config.zero_step_losses.image = params.get('image_loss_scale', 1.0)
  config.zero_step_losses.divergence = params.get('divergence_scale', 1.0)
  config.zero_step_losses.global_divergence = params.get('global_divergence_scale', 0.1)
  config.zero_step_losses.reward = params.get('reward_scale', 10.0)
  config.overshooting = params.get('overshooting', config.batch_shape[1] - 1)
  config.overshooting_losses = config.zero_step_losses.copy(_unlocked=True)
  config.overshooting_losses.reward = params.get(
      'overshooting_reward_scale', 100.0)
  del config.overshooting_losses['image']
  del config.overshooting_losses['global_divergence']
  config.optimizers = _define_optimizers(config, params)
  return config


def _training_schedule(config, params):
  config.train_steps = int(params.get('train_steps', 50000))
  config.test_steps = int(params.get('test_steps', 100))
  config.max_steps = int(params.get('max_steps', 2e7))
  config.train_log_every = config.train_steps
  config.train_checkpoint_every = None
  config.test_checkpoint_every = int(
      params.get('checkpoint_every', config.test_steps))
  config.savers = [tools.AttrDict(exclude=(r'.*_temporary.*',))]
  config.mean_metrics_every = config.train_steps // 10
  config.train_dir = os.path.join(params.logdir, 'train_episodes')
  config.test_dir = os.path.join(params.logdir, 'test_episodes')
  config.random_collects = _initial_collection(config, params)
  config.sim_collects = _active_collection(config, params)
  config.sim_summaries = tools.AttrDict(_unlocked=True)
  for task in config.tasks:
    for horizon in params.get('summary_horizons', [12]):
      name = 'summary-{}-cem-{}'.format(task.name, horizon)
      config.sim_summaries[name] = _define_simulation(
          task, config, params, horizon, 1)
  return config


def _define_optimizers(config, params):
  optimizers = tools.AttrDict(_unlocked=True)
  gradient_heads = params.get('gradient_heads', ['image', 'reward'])
  assert all(head in config.heads for head in gradient_heads)
  diagnostics = r'.*/head_(?!{})[a-z]+/.*'.format('|'.join(gradient_heads))
  kwargs = dict(
      optimizer_cls=functools.partial(tf.train.AdamOptimizer, epsilon=1e-4),
      learning_rate=params.get('learning_rate', 1e-3),
      schedule=functools.partial(tools.schedule.linear, ramp=10000),
      clipping=params.get('gradient_clipping', 1000.0))
  optimizers.main = functools.partial(
      tools.CustomOptimizer, include=r'.*', exclude=diagnostics, **kwargs)
  for name in config.heads:
    assert config.zero_step_losses.get(name), name
    # Diagnostic heads use separate optimizers to not interfere with the model.
    if name in gradient_heads:
      continue
    optimizers[name] = functools.partial(
        tools.CustomOptimizer, include=r'.*/head_{}/.*'.format(name), **kwargs)
  return optimizers


def _initial_collection(config, params):
  num_seed_episodes = params.get('num_seed_episodes', 5)
  sims = tools.AttrDict(_unlocked=True)
  for task in config.tasks:
    sims['train-' + task.name] = tools.AttrDict(
        task=task,
        save_episode_dir=config.train_dir,
        num_episodes=num_seed_episodes)
    sims['test-' + task.name] = tools.AttrDict(
        task=task,
        save_episode_dir=config.test_dir,
        num_episodes=num_seed_episodes)
  return sims


def _active_collection(config, params):
  sims = tools.AttrDict(_unlocked=True)
  batch_size = params.get('collect_batch_size', 1)
  for task in config.tasks:
    for index, horizon in enumerate(params.get('collect_horizons', [12])):
      sim = _define_simulation(task, config, params, horizon, batch_size)
      sim.unlock()
      sim.save_episode_dir = config.train_dir
      sim.steps_after = params.get('collect_every', 5000)
      sim.steps_every = params.get('collect_every', 5000)
      sim.exploration = tools.AttrDict(
          scale=params.get('exploration_noises', [0.3])[index],
          schedule=functools.partial(
              tools.schedule.linear,
              ramp=params.get('exploration_ramps', [0])[index]))
      sims['train-{}-cem-{}'.format(task.name, horizon)] = sim
      if params.get('collect_test', False):
        sim = sim.copy()
        sim.save_episode_dir = config.test_dir
        sims['test-{}-cem-{}'.format(task.name, horizon)] = sim
  return sims


def _define_simulation(task, config, params, horizon, batch_size):
  def objective(state, graph):
    return graph.heads['reward'](graph.cell.features_from_state(state)).mean()
  planner = functools.partial(
      control.planning.cross_entropy_method,
      amount=params.get('cem_amount', 1000),
      topk=params.get('cem_topk', 100),
      iterations=params.get('cem_iterations', 10),
      horizon=horizon)
  return tools.AttrDict(
      task=task,
      num_agents=batch_size,
      planner=planner,
      objective=objective)
