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

import collections
import functools
import os

import numpy as np

from planet import control
from planet import networks
from planet import tools


Task = collections.namedtuple(
    'Task', 'name, env_ctor, max_length, state_components')


def cartpole_balance(config, params):
  action_repeat = params.get('action_repeat', 8)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity']
  env_ctor = functools.partial(
      _dm_control_env, action_repeat, max_length, 'cartpole', 'balance')
  return Task('cartpole_balance', env_ctor, max_length, state_components)


def cartpole_swingup(config, params):
  action_repeat = params.get('action_repeat', 8)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity']
  env_ctor = functools.partial(
      _dm_control_env, action_repeat, max_length, 'cartpole', 'swingup')
  return Task('cartpole_swingup', env_ctor, max_length, state_components)


def finger_spin(config, params):
  action_repeat = params.get('action_repeat', 2)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity', 'touch']
  env_ctor = functools.partial(
      _dm_control_env, action_repeat, max_length, 'finger', 'spin')
  return Task('finger_spin', env_ctor, max_length, state_components)


def cheetah_run(config, params):
  action_repeat = params.get('action_repeat', 4)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity']
  env_ctor = functools.partial(
      _dm_control_env, action_repeat, max_length, 'cheetah', 'run')
  return Task('cheetah_run', env_ctor, max_length, state_components)


def cup_catch(config, params):
  action_repeat = params.get('action_repeat', 6)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity']
  env_ctor = functools.partial(
      _dm_control_env, action_repeat, max_length, 'ball_in_cup', 'catch')
  return Task('cup_catch', env_ctor, max_length, state_components)


def walker_walk(config, params):
  action_repeat = params.get('action_repeat', 2)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'height', 'orientations', 'velocity']
  env_ctor = functools.partial(
      _dm_control_env, action_repeat, max_length, 'walker', 'walk')
  return Task('walker_walk', env_ctor, max_length, state_components)


def humanoid_walk(config, params):
  action_repeat = params.get('action_repeat', 2)
  max_length = 1000 // action_repeat
  state_components = [
      'reward', 'com_velocity', 'extremities', 'head_height', 'joint_angles',
      'torso_vertical', 'velocity']
  env_ctor = functools.partial(
      _dm_control_env, action_repeat, max_length, 'humanoid', 'walk')
  return Task('humanoid_walk', env_ctor, max_length, state_components)


def _dm_control_env(action_repeat, max_length, domain, task):
  from dm_control import suite
  def env_ctor():
    env = control.wrappers.DeepMindWrapper(suite.load(domain, task), (64, 64))
    env = control.wrappers.ActionRepeat(env, action_repeat)
    env = control.wrappers.LimitDuration(env, max_length)
    env = control.wrappers.PixelObservations(env, (64, 64), np.uint8, 'image')
    env = control.wrappers.ConvertTo32Bit(env)
    return env
  env = control.wrappers.ExternalProcess(env_ctor)
  return env
