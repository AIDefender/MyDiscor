from collections import OrderedDict

import numpy as np
import os
import random

import torch

from rlkit.data_management.replay_buffer import ReplayBuffer


class SimpleReplayBuffer(ReplayBuffer):

    def __init__(
        self,
        max_replay_buffer_size,
        observation_dim,
        action_dim,
        env_info_sizes,
        path='',
        preload_buffer=False,
        priority=False,
    ):
        self._observation_dim = int(observation_dim)
        self._action_dim = int(action_dim)
        self._max_replay_buffer_size = int(max_replay_buffer_size)
        self._observations = np.zeros((self._max_replay_buffer_size, self._observation_dim))

        self.priority = priority
        if self.priority:
            self.priorities = np.zeros((self._max_replay_buffer_size, 1), dtype=np.float32)
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((max_replay_buffer_size, observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')
        # Define self._env_infos[key][i] to be the return value of env_info[key]
        # at time i
        self._env_infos = {}
        for key, size in env_info_sizes.items():
            self._env_infos[key] = np.zeros((max_replay_buffer_size, size))
        self._env_info_keys = env_info_sizes.keys()

        self._top = 0
        self._size = 0
        self._num_static_samples = 0


    def add_sample(self, observation, action, reward, next_observation,
                   terminal, env_info, **kwargs):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation

        for key in self._env_info_keys:
            self._env_infos[key][self._top] = env_info[key]
        self._advance()

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % (self._max_replay_buffer_size - self._num_static_samples)
        self._top = self._top + self._num_static_samples
        if self._size < self._max_replay_buffer_size:
            self._size += 1
    
    def random_batch(self, batch_size):
#        if self.prioritize:
#            prob_alpha = 0.6
#            probs = self.priorities[:self._size]
#            probs = probs ** prob_alpha
#            probs /= probs.sum()
#            indices = np.random.choice(self._size, batch_size, p=probs)
#        else:
        indices = np.random.randint(0, self._size, batch_size)
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch

    def rebuild_env_info_dict(self, idx):
        return {
            key: self._env_infos[key][idx]
            for key in self._env_info_keys
        }

    def batch_env_info_dict(self, indices):
        return {
            key: self._env_infos[key][indices]
            for key in self._env_info_keys
        }

    def num_steps_can_sample(self):
        return self._size

    def get_diagnostics(self):
        return OrderedDict([
            ('size', self._size)
        ])

    def end_epoch(self, epoch, save=False, log_name=''):
        if save:
            to_save = {
                    'obs': self._observations,
                    'actions': self._actions,
                    'rewards': self._rewards,
                    'terminals': self._terminals,
                    'next_obs': self._next_obs,
                    'env_infos': self._env_infos,
            }
            torch.save(to_save, os.path.join('./buffers', log_name + '.tar'))
        else:
            return
