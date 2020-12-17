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
from tactile_baselines.utils.utils import set_seed, ClassEncoder
from tactile_baselines.common.tf_util import get_session
from tactile_baselines import logger
from importlib import import_module
from pdb import set_trace as st
import pickle

INSTANCE_TYPE = 'c4.8xlarge'
EXP_NAME = 'none/2'

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

    testing_permute = extra_args['testing_permute']
    training_permute = extra_args['training_permute']

    env = build_env(args, permute = training_permute)
    eval_env = build_env(args, permute = testing_permute)


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
        eval_env=eval_env,
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


def build_env(args, permute = False):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed
    obs_type = args.obs_type
    fixed_num_of_contact = args.fixed_num_of_contact

    fixed_num_of_contact = [fixed_num_of_contact, permute]

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
    if kwargs['env'] == 'HandManipulateEgg-v0':
        kwargs['fixed_num_of_contact'] == 9
    elif kwargs['env'] == 'HandManipulateBlock-v0':
        kwargs['fixed_num_of_contact'] == 16
    elif kwargs['env'] == 'HandManipulatePen-v0':
        kwargs['fixed_num_of_contact'] == 6
    elif kwargs['env'] == 'SoftHandManipulateEgg-v0':
        kwargs['fixed_num_of_contact'] == 30

    if kwargs['pre_train_model'] in ['supervised', 'cpc']:
        if kwargs['pre_train_model'] == 'supervised':
            kwargs['feature_net_path'] = '/home/vioichigo/try/tactile-baselines/tactile_baselines/saved_model/max_pool/'
        elif kwargs['pre_train_model'] == 'cpc':
            if kwargs['process_type'] == 'max_pool':
                kwargs['feature_net_path'] = '/home/vioichigo/try/tactile-baselines/saved_model/cpc2/trained'
            elif kwargs['process_type'] == 'none':
                kwargs['feature_net_path'] = '/home/vioichigo/try/tactile-baselines/saved_model/cpc2/none/trained'



    arg_list = []
    for key in kwargs.keys():
        arg_list.append('--' + key)
        arg_list.append(str(kwargs[key]))
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(arg_list)
    extra_args = parse_cmdline_kwargs(unknown_args)

    params = args.__dict__
    import copy
    params = copy.deepcopy(params)
    params['testing_permute'] = kwargs['testing_permute']
    params['training_permute'] = kwargs['training_permute']
    params['note'] = kwargs['note']
    params['include_action'] = kwargs['include_action']
    if kwargs['pre_train_model'] == 'none':
        params['label'] = args.obs_type + '(' + args.process_type+ ')'
    else:
        if kwargs['update_model']:
            params['label'] = args.obs_type + '(' + kwargs['pre_train_model']+ ')'
        else:
            params['label'] = args.obs_type + '(' + kwargs['pre_train_model']+ ', no update)'

    exp_dir = os.getcwd() + '/data/' + EXP_NAME
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last')
    json.dump(params, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = kwargs.get('gpu_frac', 0.95)
    sess = tf.Session(config=config)
    if kwargs['save_policy']:
        extra_args['save_path'] = exp_dir
    if kwargs['load_policy']:
        extra_args['load_path'] = exp_dir
    extra_args['path'] = exp_dir
    model, env = train(args, extra_args)

    return model

if __name__ == '__main__':
    sweep_params = {
        'alg': ['her'],
        'seed': [22],
        'env': ['HandManipulateEgg-v0'],

        # Env Sampling
        'num_timesteps': [1e6],
        'fixed_num_of_contact': [9],

        # Problem Conf
        'num_cpu': [19],
        # baseline: 'object_loc+rot+other' 61
        # no object location: 'rot+other' 58
        # full_contact: 'object_loc+rot+geom+contact_loc+force+other' 583
        # rot_contact: 'rot+geom+contact_loc+force+other' 580
        # no geoms: 'rot+contact_loc+force+other' 139
        # no force: 'rot+geom+contact_loc+other' 526
        # only loc: 'rot+contact_loc+other' 85
        # 'rot+geom+contact_loc+force+other', 'rot+contact_loc+force+other', 'rot+contact_loc+other'
        'obs_type': ['object_loc+rot+geom+contact_loc+force+other'],
        # 'obs_type': ['object_loc+rot+other'],
        'process_type': ['none'], #none, max_pool
        'feature_dim': [256],
        'pre_train_model': ['none'], #'supervised', 'cpc', 'none'
        'include_action': [False], #for cpc only
        'feature_net_path': [''],
        'update_model': [False],
        'testing_permute': [False],
        'training_permute': [False],
        'save_policy': [False],
        'load_policy': [False],
        'note': ['try'],
        'proj_name': ['tactile-debug'],
        }
    run_sweep(main, sweep_params, EXP_NAME, INSTANCE_TYPE)

    # python run_scripts/her_run_sweep.py
