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
import tactile_baselines.her.experiment.config as configure
from tactile_baselines.her.experiment.config import configure_her

from tactile_baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from tactile_baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from tactile_baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from tactile_baselines.common.tf_util import get_session
from tactile_baselines.common.replay_buffers import SequenceReplayBuffer
from run_scripts.functional.rollout import RolloutWorker
from tactile_baselines import logger

from importlib import import_module
from pdb import set_trace as st
import dill as pickle

INSTANCE_TYPE = 'c4.8xlarge'
EXP_NAME = 'supervised'

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
        fixed_num_of_contact = [fixed_num_of_contact, False],
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
                       fixed_num_of_contact = [fixed_num_of_contact, False])

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
    buffer_size = int(kwargs['buffer_size'])
    args, unknown_args = arg_parser.parse_known_args(arg_list)
    extra_args = parse_cmdline_kwargs(unknown_args)

    params = args.__dict__
    import copy
    params = copy.deepcopy(params)

    if args.obs_type == 'object':
        params['label'] = args.obs_type
    elif args.obs_type == 'original':
        params['label'] = 'object+joint'
    elif args.obs_type == 'contact':
        params['label'] = 'object+contact(' + args.process_type + ')'
    elif args.obs_type == 'full_contact':
        params['label'] = 'object+joint+contact(' + args.process_type + ')'


    exp_dir = os.getcwd() + '/data/' + EXP_NAME
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last')
    json.dump(params, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = kwargs.get('gpu_frac', 0.95)
    sess = tf.Session(config=config)

    obs_type = params['obs_type']
    fixed_num_of_contact = params['fixed_num_of_contact']

    env = gym.make(params['env'], obs_type = obs_type, fixed_num_of_contact = [fixed_num_of_contact, False])

    policy_type = kwargs['policy']
    policy = pickle.load(open('../policy/' + str(args.env) + '/policy.pickle', 'rb'))

    T = env._max_episode_steps


    paths = generate_paths(policy,
                           T,
                           obs_type,
                           params['env'],
                           fixed_num_of_contact,
                           build_env(args),
                           contact_dim = env.contact_dim,
                           buffer_size = buffer_size,
                           policy_type = policy_type)

    obs, act = paths.all_samples()
    obs = process_episode(obs, env.contact_dim, fixed_num_of_contact)

    folder = '../dataset/sequence/' + str(args.env) + '/seed' + str(kwargs['seed'])
    with open(folder + '-dict.pickle', 'wb') as pickle_file:
        print(folder)
        pickle.dump([obs, act, fixed_num_of_contact], pickle_file)


def process_episode(observations, contact_dim, fixed_num_of_contact):
    n, epi_length, obs_dim = observations.shape
    observations = observations.reshape((-1, obs_dim))
    contacts = observations[:, :contact_dim]
    single_dim = contact_dim // fixed_num_of_contact
    contacts = contacts.reshape((-1, fixed_num_of_contact, single_dim))
    empty = -np.ones((contacts.shape[0], contacts.shape[1], 2))
    indices = np.transpose((contacts[:, :, :-9]==1.0).nonzero())
    rows, cols, vals = tuple(indices[:, 0][::2]), tuple(indices[:, 1][::2]), indices[:, 2]
    empty[rows, cols] = vals.reshape((-1,2))
    other_information = observations[:, contact_dim:].reshape((contacts.shape[0], -1))
    transformed_dict = {}
    transformed_contacts = np.concatenate((empty, contacts[:, :, -9:]), axis = -1)
    transformed_dict['geom1s'] = np.expand_dims(empty[:, :, 0], axis = -1) #N*num_contact*1
    transformed_dict['geom2s'] = np.expand_dims(empty[:, :, 1], axis = -1) #N*num_contact*1
    transformed_dict['positions'] =  contacts[:, :, -9:-6] #N*num_contact*3
    transformed_dict['force'] =  contacts[:, :, -6:] #N*num_contact*6
    # dimension: 24:24:6:7
    transformed_dict['object_position'] =  other_information[:, 48+6:] #N*7
    transformed_dict['object_vel'] =  other_information[:, 48:48+6] #N*6
    transformed_dict['joint_position'] =  other_information[:, :24] #N*24
    transformed_dict['joint_vel'] =  other_information[:, 24:48] #N*24
    for key in transformed_dict:
        if len(transformed_dict[key].shape) == 2:
            _, dim = transformed_dict[key].shape
            transformed_dict[key] = transformed_dict[key].reshape((n, epi_length, dim))
        elif len(transformed_dict[key].shape) == 3:
            _, num, dim = transformed_dict[key].shape
            transformed_dict[key] = transformed_dict[key].reshape((n, epi_length, num, dim))
    return transformed_dict

def generate_paths(policy,
                   T,
                   obs_type,
                   env_name,
                   fixed_num_of_contact,
                   env,
                   contact_dim,
                   buffer_size,
                   policy_type):
    rollout_params = {
        'exploit': True,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'T': T,
        'contact_dim': contact_dim,
    }

    params = configure.DEFAULT_PARAMS
    env_name = env.spec.id
    params['env_name'] = env_name
    replay_strategy = 'future'
    params['replay_strategy'] = replay_strategy
    params['obs_type'] = obs_type
    params['fixed_num_of_contact'] = fixed_num_of_contact
    if env_name in configure.DEFAULT_ENV_PARAMS:
        params.update(configure.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params = configure.prepare_params(params)
    params['rollout_batch_size'] = env.num_envs

    dims = configure.configure_dims(params)
    # dims['o'] = policy.dimo
    rollout_worker = RolloutWorker(env,
                                   policy,
                                   dims,
                                   logger,
                                   monitor=True,
                                   noise_eps = 0.1,
                                   random_eps = 0.1,
                                   use_random_policy = (policy_type == 'random'),
                                   **rollout_params)
    rollout_worker.clear_history()
    obs_dim = int(np.prod(env.observation_space['observation'].shape))
    act_dim = int(np.prod(env.action_space.shape))
    buffer = SequenceReplayBuffer(obs_dim, act_dim, buffer_size, episode_size = 100)

    while buffer._size < buffer_size:
        print(buffer._size, buffer_size)
        episode = rollout_worker.generate_rollouts()
        # (8.837223, -10.698544, 0.029151862) (0.9997292357316148, -0.9992342112650581, 0.014519963432654456)
        # (11.591648, -14.190264, 0.03476813) (0.9978768825193871, -0.9985089559162041, 0.006812978986824379)
        full_obs = episode['o'].reshape((-1, obs_dim)) #100, 583
        acts = episode['u'].reshape((-1, act_dim))
        buffer.add_samples(full_obs, acts)
    return buffer


if __name__ == '__main__':
    sweep_params = {
        'alg': ['her'],
        'seed': [2],
        'env': ['HandManipulateEgg-v0'],

        # Env Sampling
        'num_timesteps': [1e6],
        'fixed_num_of_contact': [9],
        'buffer_size': [1e5//2],

        # Problem Conf
        'num_cpu': [1],
        'obs_type': ['object_loc+rot+geom+contact_loc+force+other'],
        'process_type': ['none'],
        'feature_dim': [32],
        'policy': ['trained'], #'random', 'trained'
        }
    run_sweep(main, sweep_params, EXP_NAME, INSTANCE_TYPE)
    # python run_scripts/functional/generate_dataset.py
