import numpy as np

from tactile_baselines.utils.serializable import Serializable

from .replay_buffer import ReplayBuffer
from pdb import set_trace as st
import copy


class SequenceReplayBuffer(ReplayBuffer, Serializable):
    def __init__(self, obs_dim, act_dim, max_replay_buffer_size, episode_size = 100):
        super(SequenceReplayBuffer, self).__init__()
        Serializable.quick_init(self, locals())

        max_replay_buffer_size = int(max_replay_buffer_size)

        self._observation_dim = obs_dim
        self._action_dim = act_dim
        self._max_buffer_size = max_replay_buffer_size
        self._episode_size = episode_size
        self._neps = (max_replay_buffer_size // episode_size) + 1
        self._observations = np.zeros((self._neps, episode_size, self._observation_dim))
        self._actions = np.zeros((self._neps, episode_size-1, self._action_dim))
        self._size = 0
        self._episode = 0

    def add_samples(self, observations, actions, **kwargs):
        assert len(observations) == len(actions) + 1
        total_num = len(observations)
        if total_num != self._episode_size:
            return
        observations, actions = copy.deepcopy([observations, actions])

        self._observations[self._episode] = observations
        self._actions[self._episode] = actions

        self._advance(num=total_num)

    def add_sample(self, observation, action, reward, terminal, **kwargs):
        return


    def all_samples(self):
        return self._observations[:self._episode], self._actions[:self._episode]

    def _advance(self, num=1):
        self._episode += 1
        self._size += num

    def random_batch(self, batch_size, prefix=''):
        indices = np.random.randint(0, self._episode)
        result = dict()
        result[prefix + 'observations'] = self._observations[indices].copy()
        result[prefix + 'actions'] = self._actions[indices].copy()
        return result


    def terminate_episode(self):
        pass

    @property
    def size(self):
        return self._size
