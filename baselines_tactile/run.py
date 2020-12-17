import sys
import re
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import os
import json
import tensorflow as tf
import numpy as np
from meta_mb.utils.utils import ClassEncoder

from tactile_baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from tactile_baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from tactile_baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from tactile_baselines.common.tf_util import get_session
from tactile_baselines import logger
from importlib import import_module
from pdb import set_trace as st

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
    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"), record_video_trigger=lambda x: x % args.save_video_interval == 0, video_length=args.save_video_length)

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

    # if env_type in {'atari', 'retro'}:
    #     # if alg == 'deepq':
    #     #     env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
    #     # elif alg == 'trpo_mpi':
    #     #     env = make_env(env_id, env_type, seed=seed)
    #     # else:
    #     frame_stack_size = 4
    #     env = make_vec_env(env_id, env_type, nenv, seed, gamestate=args.gamestate, reward_scale=args.reward_scale)
    #     env = VecFrameStack(env, frame_stack_size)
    #
    # else:
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


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)


def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)

    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)
    if args.debug:
        exp_dir = os.getcwd() + '/data/debug'
    elif args.log_path:
        exp_dir = os.getcwd() + '/data/' + args.env + '/' + args.log_path + '-' + args.obs_type + '-' + args.process_type + '-' + str(args.feature_dim)
    else:
        exp_dir = os.getcwd() + '/data/' + args.env  + '/' + args.obs_type + '-' + args.process_type + '-' + str(args.feature_dim)

    # add one more to args
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        # configure_logger(args.log_path)
        logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='gap', snapshot_gap=5)
        json.dump(args.__dict__, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        # configure_logger(args.log_path, format_strs=[])
        logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='gap', snapshot_gap=5)
        json.dump(args.__dict__, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)

    model, env = train(args, extra_args)

    save_path = osp.expanduser(exp_dir) + '/policy.pickle'
    print(save_path)
    model.save(save_path)

    return model

if __name__ == '__main__':
    main(sys.argv)
