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
import tactile_baselines.her.experiment.config as config
from tactile_baselines.her.experiment.config import configure_her

from tactile_baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from tactile_baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from tactile_baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from tactile_baselines.common.tf_util import get_session
from tactile_baselines.common.replay_buffers import SequenceReplayBuffer
from tactile_baselines import logger

from importlib import import_module
from tactile_baselines.cpc.data_util import *
from pdb import set_trace as st
import dill as pickle

INSTANCE_TYPE = 'c4.8xlarge'
EXP_NAME = 'supervised'


def main(**kwargs):
    # configure logger, disable logging in child MPI processes (with rank > 0)

    # folder = '../dataset/sequence/HandManipulateEgg-v0/seed'
    # obs = {}
    # acs = []
    # for i in range(2):
    #     # with open(folder + str(i) + '-dict.pickle', 'wb') as pickle_file:
    #     # dict, array, int
    #     o, a, fixed_num_of_contact = pickle.load(open(folder + str(i+1) + '-dict.pickle', 'rb'))
    #     for key in o:
    #         if key in obs:
    #             obs[key] = np.concatenate([obs[key], o[key]], axis = 0)
    #         else:
    #             obs[key] = o[key]
    #     acs.append(a)
    # acs = np.concatenate(acs, axis = 0)
    # folder = './dataset/sequence/HandManipulateEgg-v0/2seeds'
    # with open(folder + '-dict.pickle', 'wb') as pickle_file:
    #     print(folder)
    #     filtered_obs = {}
    #     for key in obs:
    #         if key in ['geom1s', 'geom2s', 'positions', 'force', 'object_position']:
    #             filtered_obs[key] = obs[key]
    #
    #     pickle.dump([filtered_obs, acs, fixed_num_of_contact], pickle_file)
    #
    # # ../sequence/HandManipulateEgg-v09/5seeds-dict.pickle
    folder = '../dataset/sequence/HandManipulateEgg-v0/seed1-dict.pickle'
    o, a, fixed_num_of_contact = pickle.load(open(folder, 'rb'))
    env = gym.make(kwargs['env'],
                   obs_type = kwargs['obs_type'],
                   fixed_num_of_contact = [fixed_num_of_contact, True])

    ngeoms = env.sim.model.ngeom
    obs, object_info = expand_data(o, ngeoms, fixed_num_of_contact)
    folder = './dataset/HandManipulateEgg-v0/50000obs.pickle'
    obs = obs.reshape((-1, *obs.shape[2:]))
    with open(folder, 'wb') as pickle_file:
        pickle.dump(obs, pickle_file)





if __name__ == '__main__':
    sweep_params = {
        'alg': ['her'],
        'seed': [6],
        'env': ['HandManipulateEgg-v0'],

        # Env Sampling
        'num_timesteps': [1e6],
        'fixed_num_of_contact': [9],
        'buffer_size': [1e6],

        # Problem Conf
        'num_cpu': [1],
        'obs_type': ['object_loc+rot+geom+contact_loc+force+other'],
        'process_type': ['none'],
        'feature_dim': [32],
        }
    run_sweep(main, sweep_params, EXP_NAME, INSTANCE_TYPE)
    # python run_scripts/functional/merge.py
