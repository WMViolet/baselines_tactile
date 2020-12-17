from envs.gym.envs.mujoco.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from envs.gym.envs.mujoco.ant import AntEnv
from envs.gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from envs.gym.envs.mujoco.hopper import HopperEnv
from envs.gym.envs.mujoco.walker2d import Walker2dEnv
from envs.gym.envs.mujoco.humanoid import HumanoidEnv
from envs.gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from envs.gym.envs.mujoco.inverted_double_pendulum import InvertedDoublePendulumEnv
from envs.gym.envs.mujoco.reacher import ReacherEnv
from envs.gym.envs.mujoco.swimmer import SwimmerEnv
from envs.gym.envs.mujoco.humanoidstandup import HumanoidStandupEnv
from envs.gym.envs.mujoco.pusher import PusherEnv
from envs.gym.envs.mujoco.thrower import ThrowerEnv
from envs.gym.envs.mujoco.striker import StrikerEnv
