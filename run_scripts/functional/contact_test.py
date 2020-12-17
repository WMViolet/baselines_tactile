import sys
import re
import multiprocessing
import os.path as osp
from envs import gym
from collections import defaultdict
import os
import json
import tensorflow as tf
import numpy as np
from experiment_utils.run_sweep import run_sweep

from tactile_baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from tactile_baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from tactile_baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from tactile_baselines.common.tf_util import get_session
from tactile_baselines import logger
from importlib import import_module
from pdb import set_trace as st
import pickle

INSTANCE_TYPE = 'c4.8xlarge'
EXP_NAME = 'joint'

class ClassEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        if callable(o):
            return {'function': o.__name__}
        return json.JSONEncoder.default(self, o)

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


def train(args, extra_args):
    env_type, env_id = get_env_type(args)
    print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed
    num_cpu = args.num_cpu

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = build_env(args)

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))
    obs_type = args.obs_type
    process_type = args.process_type
    feature_dim = args.feature_dim
    fixed_num_of_contact = args.fixed_num_of_contact

    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        num_cpu = num_cpu,
        obs_type = obs_type,
        process_type = process_type,
        feature_dim = feature_dim,
        fixed_num_of_contact = fixed_num_of_contact,
        **alg_kwargs
    )

    return model, env


def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed
    obs_type = args.obs_type
    fixed_num_of_contact = args.fixed_num_of_contact

    env_type, env_id = get_env_type(args)

    config = tf.ConfigProto(allow_soft_placement=True,
                           intra_op_parallelism_threads=1,
                           inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    get_session(config=config)

    flatten_dict_observations = alg not in {'her'}

    env = make_vec_env(env_id,
                       env_type,
                       args.num_env or 1,
                       seed,
                       reward_scale=args.reward_scale,
                       flatten_dict_observations=flatten_dict_observations,
                       obs_type = obs_type,
                       fixed_num_of_contact = fixed_num_of_contact)

    if env_type == 'mujoco':
        env = VecNormalize(env, use_tf=True)

    return env


def get_env_type(args):
    env_id = args.env

    if args.env_type is not None:
        return args.env_type, env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['tactile_baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs



def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}



def main(**kwargs):
    # configure logger, disable logging in child MPI processes (with rank > 0)
    arg_list = []
    for key in kwargs.keys():
        arg_list.append('--' + key)
        arg_list.append(str(kwargs[key]))
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(arg_list)
    extra_args = parse_cmdline_kwargs(unknown_args)

    params = args.__dict__

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = kwargs.get('gpu_frac', 0.95)
    sess = tf.Session(config=config)
    env = gym.make(kwargs['env'],
                   obs_type = kwargs['obs_type'],
                   fixed_num_of_contact = kwargs['fixed_num_of_contact'])


    global_largest = 0
    for _ in range(200):
        horizon = 100
        o = env.reset()
        d = False
        t = 0
        largest = 0
        while t < horizon and d is False:
            a = env.action_space.sample()
            num_of_contacts = env.contact_num
            if num_of_contacts > largest:
                largest = num_of_contacts
            if largest > global_largest:
                global_largest = largest
            o, r, d, _ = env.step(a)
            t = t+1
        print("largest: ", largest)
    print("global_largest: ", global_largest)
    env.close()



if __name__ == '__main__':
    sweep_params = {
        'alg': ['her'],
        'seed': [1],
        'env': ['SoftHandManipulateEgg-v0'],

        # Env Sampling
        'num_timesteps': [1e6],
        'fixed_num_of_contact': [50],

        # Problem Conf
        'num_cpu': [1],
        'obs_type': ['full_contact'],
        'process_type': ['max_pool'],
        'feature_dim': [32],
        }
    run_sweep(main, sweep_params, EXP_NAME, INSTANCE_TYPE)
