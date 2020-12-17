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
from supervised.rotation_model import RotationModel
import torch
from tactile_baselines import logger
from tactile_baselines.utils.utils import set_seed, ClassEncoder
from scipy.spatial.transform import Rotation as R
from pdb import set_trace as st
import dill as pickle
from datetime import datetime
from utils.utils import *
from utils.mlp import *


def main(**kwargs):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = kwargs.get('gpu_frac', 0.95)
    sess = tf.Session(config=config)
    exp_dir = os.getcwd() + '/data/feature_net/' + kwargs['input_label'][0] + kwargs['output_label'][0] + '/'
    mode = kwargs['mode'][0]

    if mode == 'restore':
        rotation_saver = tf.train.import_meta_graph(exp_dir + '-999.meta')
        rotation_saver.restore(sess, tf.train.latest_checkpoint(exp_dir))
        graph = tf.get_default_graph()

    with sess.as_default() as sess:

        input_label = kwargs['input_label'][0]
        output_label = kwargs['output_label'][0]
        buffer = {}
        name = '1'
        paths, fixed_num_of_contact = pickle.load(open('../saved/trained/SoftHandManipulateEgg-v080-' + name + '-dict.pickle', 'rb'))
        for key in paths:
            buffer[key] = paths[key]

        for name in [str(i) for i in range(2, 17)]:
            paths, fixed_num_of_contact = pickle.load(open('../saved/trained/SoftHandManipulateEgg-v080-' + name + '-dict.pickle', 'rb'))
            for key in paths:
                buffer[key] = np.concatenate([buffer[key], paths[key]], axis = 0)


        env = gym.make(kwargs['env'][0],
                       obs_type = kwargs['obs_type'][0],
                       fixed_num_of_contact = fixed_num_of_contact)
        batch_size = 100
        paths = data_filter(buffer, fixed_num_of_contact, batch_size)
        niters = paths['positions'].shape[0] // batch_size
        print("total iteration: ", niters)
        print("total number of data: ", paths['positions'].shape[0])

        train_data, test_data, _, _ = split_data(paths, niters)
        train_data['object_position'] = train_data['object_position'][:, :, :3]
        test_data['object_position'] = test_data['object_position'][:, :, :3]

        labels_to_dims = {}
        labels_to_dims['positions'] = 3


        rotation_model = RotationModel(dims = [labels_to_dims[input_label]],
                                       sess = sess,
                                       fixed_num_of_contact = fixed_num_of_contact,
                                       feature_layers = kwargs['feature_layers'][0],
                                       output_layers = kwargs['output_layers'][0],
                                       learning_rate = kwargs['learning_rate'][0])

        if mode == 'train':
            sess.run(tf.global_variables_initializer())
            for i in range(niters):
                input, out = train_data[input_label][i], train_data[output_label][i]
                pred = rotation_model.train(input, out)
                logger.logkv("iter", i)
                logger.dumpkvs()
            rotation_model.save_model(exp_dir, 999)

        if mode == 'restore':
            rotation_model.restore()
            for i in range(1):
                logger.logkv("iter", i)
                _, _ = rotation_model.restore_predict(train_data[input_label][i], train_data[output_label][i])
                logger.dumpkvs()



if __name__ == '__main__':
    sweep_params = {
        'alg': ['her'],
        'seed': [1],
        'env': ['SiteHandManipulateEgg-v0'],

        # Env Sampling
        'num_timesteps': [1e6],
        'buffer_size': [1e6],

        # Problem Conf
        'obs_type': ['full_contact'],
        'input_label': ['positions'],
        'output_label': ['rotations'],
        'mode': ['restore'],
        'feature_layers': [[32, 32]],
        'output_layers': [[32, 32]],
        'visualize_training_data': [False],
        'visualize_testing_data': [False],
        'learning_rate': [1e-3],
        }
    main(**sweep_params)
