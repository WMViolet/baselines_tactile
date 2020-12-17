import numpy as np

from tactile_baselines.utils.serializable import Serializable

from .replay_buffer import ReplayBuffer
from pdb import set_trace as st
import copy


class ObsReplayBuffer(ReplayBuffer, Serializable):
    def __init__(self, obs_dim, object_dim, max_replay_buffer_size):
        super(ObsReplayBuffer, self).__init__()
        Serializable.quick_init(self, locals())

        max_replay_buffer_size = int(max_replay_buffer_size)

        self._observation_dim = obs_dim
        self._object_dim = object_dim
        self._max_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size,
                                       self._observation_dim))
        self._top = 0
        self._size = 0

    def add_sample(self, observation, **kwargs):
        self._observations[self._top] = observation.copy()

        self._advance()

    def add_samples(self, observations, **kwargs):
        total_num = observations.shape[0]
        observations = copy.deepcopy([observations])[0]
        if self._top + total_num <= self._max_buffer_size:
            self._observations[self._top: self._top + total_num] = observations
        else:
            back_size = self._max_buffer_size - self._top
            redundant = (total_num - back_size) // self._max_buffer_size
            remaining = (total_num - back_size) % self._max_buffer_size
            if redundant == 0:
                self._observations[self._top:] = observations[:back_size]

                self._observations[:total_num - back_size] = observations[back_size:]

            else:
                print("WARNING: there are ", redundant * self._max_buffer_size, " samples that are not used. ")
                self._observations[:] = observations[back_size + (redundant - 1) * self._max_buffer_size: back_size + redundant * self._max_buffer_size]
                self._observations[:remaining] = observations[back_size + redundant * self._max_buffer_size:]

        self._advance(num=total_num)


    def all_samples(self):
        return self._observations[:self._size]

    def terminate_episode(self):
        pass

    def _advance(self, num=1):
        self._top = (self._top + num) % self._max_buffer_size
        if self._size + num < self._max_buffer_size:
            self._size += num
        else:
            self._size = self._max_buffer_size

    def random_batch(self, batch_size, prefix=''):
        indices = np.random.randint(0, self._size, batch_size)
        result = dict()
        result[prefix + 'observations'] = self._observations[indices].copy()
        return result

    def random_batch_simple(self, batch_size, prefix = ''):
        indices = np.random.randint(0, self._size, batch_size)
        result = dict()
        result[prefix + 'observations'] = self._observations[indices].copy()
        return result

    @property
    def size(self):
        return self._size
