from .dclaw import *
from .baoding import *
from .cube import *


from envs.gym.envs.registration import register

register(
    id='pddm_baoding-v0',
    entry_point='envs.hand_envs.baoding.baoding_env:BaodingEnv',
    max_episode_steps=1000,
)

register(
    id='pddm_cube-v0',
    entry_point='envs.hand_envs.cube.cube_env:CubeEnv',
    max_episode_steps=1000,
)
