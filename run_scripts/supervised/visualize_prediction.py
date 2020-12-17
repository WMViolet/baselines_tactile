import sys
import re
import os.path as osp
from envs import gym
from collections import defaultdict
import os
import json
import tensorflow as tf
import numpy as np
import time

from tactile_baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from tactile_baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from tactile_baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from tactile_baselines.common.tf_util import get_session
from tactile_baselines.her.rollout import RolloutWorker
from tactile_baselines import logger

from importlib import import_module
from pdb import set_trace as st
import dill as pickle


INSTANCE_TYPE = 'c4.8xlarge'
EXP_NAME = 'supervised'



def main(**kwargs):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = kwargs.get('gpu_frac', 0.95)
    sess = tf.Session(config=config)


    env = gym.make(kwargs['env'][0],
                   obs_type = kwargs['obs_type'][0],
                   fixed_num_of_contact = kwargs['fixed_num_of_contact'][0])

    num_episodes = 1
    horizon = 100

    feature_net = pickle.load(open('./saved/' + str(kwargs['env'][0]) + '-model.pickle', 'rb'))

    for _ in range(num_episodes):
        o = env.reset()
        d = False
        t = 0
        while t < horizon and d is False:
            a = env.action_space.sample()
            o, r, d, _ = env.step(a)
            env.render()
            time.sleep(1)
            prediction = feature_net.predict_single(o['observation'])
            t = t+1
            env.sim.data.set_joint_qpos('object:joint', prediction[-7:])
            env.sim.forward()
            env.render()
            time.sleep(1)
            env.sim.data.set_joint_qpos('object:joint', o['observation'][-7:])
            env.sim.forward()
        env.close()
    




if __name__ == '__main__':
    sweep_params = {
        'alg': ['her'],
        'seed': [1],
        'env': ['HandManipulateEgg-v0'],

        # Env Sampling
        'num_timesteps': [1e6],
        'fixed_num_of_contact': [7],
        'buffer_size': [1e6],

        # Problem Conf
        'num_cpu': [1],
        'obs_type': ['full_contact'],
        'process_type': ['pointnet'],
        'feature_dim': [32],
        'feature_layer': [2],
        }
    main(**sweep_params)
