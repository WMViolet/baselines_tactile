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

def main(**kwargs):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = kwargs.get('gpu_frac', 0.95)
    sess = tf.Session(config=config)
    env = gym.make(kwargs['env'][0],
                   obs_type = kwargs['obs_type'][0],
                   fixed_num_of_contact = [kwargs['fixed_num_of_contact'][0], False])

    num_episodes = 100
    horizon = 100
    horizon = 1
    fixed_num_of_contact = kwargs['fixed_num_of_contact'][0]
    mode = kwargs['mode'][0]

    if mode == 'load':
        epoch = 10
        obs, predictions = pickle.load(open('./dataset/supervised-saved/' + str(epoch) + '.pickle', 'rb'))

        contact_num = fixed_num_of_contact
        B, D = obs.shape
        contact_info = obs[:, :env.contact_dim].reshape((B, fixed_num_of_contact, -1))
        object_info = obs[:, env.contact_dim:]

        for contact, o, pred in zip(contact_info, object_info, predictions):
            # env.sim.data.set_joint_qpos('target:joint', o[-7:]+np.ones(7))
            object_position = o[-7:]
            object_vel = o[48:48+6]
            joint_position = o[:24]
            joint_vel = o[24:48]
            env.sim.data.set_joint_qpos('object:joint', object_position)
            env.sim.data.set_joint_qvel('object:joint', object_vel)
            for idx in range(len(env.sim.model.joint_names)):
                name = env.sim.model.joint_names[idx]
                if name.startswith('robot'):
                    env.sim.data.set_joint_qpos(name, joint_position[idx])
                    env.sim.data.set_joint_qvel(name, joint_vel[idx])
            pos = object_position[:-4]
            num_points = (np.sum(contact, axis = 1)!=0).sum()
            if num_points != 0:
            	print(((pos - pred)**2).sum(), np.abs(pos - pred).sum(), num_points)
            else:
            	print(((pos - pred)**2).sum(), np.abs(pos - pred).sum(), pos)

            env.render()
            time.sleep(0.2)
            env.sim.data.set_joint_qpos('object:joint', np.concatenate((pred, object_position[-4:]), axis = -1))
            env.render()
            time.sleep(1)
        env.close()

        # # # note: 46 is object
	       #  for contact_idx in range(contact_num):
	       #      site_name = 'contact{}'.format(contact_idx+1)
	       #      site_id = env.sim.model.site_name2id(site_name)
	       #      env.sim.model.site_pos[site_id] = contact_info[contact_idx][-9:-6]
	       #      env.sim.forward()
	       #      time.sleep(1)
	       #  env.render()
	       #  st()

    else:
        for _ in range(num_episodes):
            o = env.reset()
            d = False
            t = 0
            while t < horizon and d is False:
                a = env.action_space.sample()
                o, r, d, _ = env.step(a)
                env.render()

                t = t+1
                # contacts = o['observation'][:env.contact_dim].reshape((fixed_num_of_contact, -1))
                # # env.sim.data.set_joint_qpos('target:joint', o['observation'][-7:]+np.ones(7))
                # contact_num = env.contact_num
                # # for idx in range(env.sim.model.ngeom):
                # #     print(idx, env.sim.model.geom_id2name(idx))
                # # note: 46 is object

                # print(contact_num)
                # for contact_idx in range(contact_num):
                #     site_name = 'contact{}'.format(contact_idx+1)
                #     site_id = env.sim.model.site_name2id(site_name)
                #     env.sim.model.site_pos[site_id] = contacts[contact_idx][-9:-6]
                #     env.sim.forward()
                #     env.render()
                #     time.sleep(1)

                # for contact_idx in range(contact_num):
                #     site_name = 'contact{}'.format(contact_idx+1)
                #     site_id = env.sim.model.site_name2id(site_name)
                #     env.sim.model.site_pos[site_id] = np.array([1, 0.9, 0.25])
                #     env.sim.forward()
                env.render()
                time.sleep(0.1)
            env.close()





if __name__ == '__main__':
    sweep_params = {
        'alg': ['her'],
        'seed': [1],
        'env': ['HandManipulateEgg-v0'],

        # Env Sampling
        'num_timesteps': [1e6],
        'fixed_num_of_contact': [9],
        'buffer_size': [1e6],

        # Problem Conf
        'obs_type': ['object_loc+rot+geom+contact_loc+force+other'],
        'mode': ['load'],
        }
    main(**sweep_params)

    # python run_scripts/functional/visualize_env.py
