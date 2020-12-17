from envs import gym
from collections import defaultdict
import os
import json
import tensorflow as tf
import numpy as np
import time

from tactile_baselines.her.supervised.model import FeatureNet
from tactile_baselines import logger
from tactile_baselines.utils.utils import set_seed, ClassEncoder
from pdb import set_trace as st
from utils.utils import *
import dill as pickle
import numpy as np



def main(**kwargs):
    exp_dir = os.getcwd() + '/tactile_baselines/saved_model/' + kwargs['process_type'][0] + '/' + str(kwargs['seed'][0]) + '/'
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last')
    json.dump(kwargs, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = kwargs.get('gpu_frac', 0.95)
    sess = tf.Session(config=config)

    mode = kwargs['mode'][0]
    batch_size = kwargs['batch_size'][0]
    input_label = kwargs['input_label'][0]
    output_label = kwargs['output_label'][0]

    if mode == 'restore':
        saver = tf.train.import_meta_graph(exp_dir + '-999.meta')
        saver.restore(sess, tf.train.latest_checkpoint(exp_dir))
        graph = tf.get_default_graph()

    with sess.as_default() as sess:
        # buffer, fixed_num_of_contact = pickle.load(open('../saved/HandManipulateEgg-v0-fix9.pickle', 'rb'))
        buffer, _, fixed_num_of_contact = pickle.load(open('../dataset/sequence/HandManipulateEgg-v0/5seeds-dict.pickle', 'rb'))

        for key in buffer:
            temp = buffer[key].copy()
            buffer[key] = temp.reshape((-1, *temp.shape[2:]))


        env = gym.make(kwargs['env'][0],
                       obs_type = kwargs['obs_type'][0],
                       fixed_num_of_contact = [fixed_num_of_contact, False])

        ngeoms = env.sim.model.ngeom
        paths = data_filter(buffer, fixed_num_of_contact, batch_size, min_num_points = 3)
        niters = paths['positions'].shape[0] // batch_size
        print("total iteration: ", niters)
        paths = expand_data(paths, ngeoms, fixed_num_of_contact)
        print("total number of data: ", paths['positions'].shape[0])

        train_data, test_data = split_data(paths, niters)
        for key in train_data:
            print(key, train_data[key].shape)

        labels_to_dims = {}
        labels_to_dims['positions'] = 3
        labels_to_dims['object_position'] = 3
        labels_to_dims['joint_position'] = 24
        labels_to_dims['geoms'] = ngeoms
        labels_to_dims['contacts'] = ngeoms + 9

        dims = (labels_to_dims[input_label], labels_to_dims[output_label])
        process_type = kwargs['process_type'][0]
        position_layers = kwargs['position_layers'][0]
        learning_rate = kwargs['learning_rate'][0]

        feature_net = FeatureNet(dims,
                                 fixed_num_of_contact = fixed_num_of_contact,
                                 sess = sess,
                                 process_type = process_type,
                                 position_layers = position_layers,
                                 learning_rate = learning_rate)

        if mode == 'train':
            sess.run(tf.global_variables_initializer())
            logger.log("training started")
            for i in range(niters):
                start = time.time()
                feature_net.train(train_data[input_label][i], train_data[output_label][i])
                feature_net.test(test_data[input_label][i], test_data[output_label][i])
                logger.logkv("iter", i)
                logger.dumpkvs()
            feature_net.save_model(exp_dir, 999)

            print("logged to", exp_dir)

            # with open(exp_dir + 'params.pickle', 'wb') as pickle_file:
            #     pickle.dump([fixed_num_of_contact, dims, position_layers, learning_rate], pickle_file)
            #
            #
            # with open(exp_dir + 'data.pickle', 'wb') as pickle_file:
            #     pickle.dump([test_data[input_label][i], test_data[output_label][i]], pickle_file)

        if mode == 'restore':
            for i in range(1):
                feature_net.restore()
                logger.logkv("iter", i)
                feature_net.test(train_data[input_label][i], train_data[output_label][i])
                logger.dumpkvs()
                with open(exp_dir + 'layers.pickle', 'wb') as pickle_file:
                    pickle.dump(position_layers, pickle_file)



if __name__ == '__main__':
    sweep_params = {
        'alg': ['her'],
        'seed': [3],
        'env': ['HandManipulateEgg-v0'],

        # Env Sampling
        'num_timesteps': [1e6],
        'buffer_size': [1e6],

        # Problem Confs
        'obs_type': ['full_contact'],
        'process_type': ['max_pool'],
        'input_label': ['positions'],
        'output_label': ['object_position'],
        'position_layers': [[[64, 32], [32, 32]]],
        'learning_rate': [1e-3],
        'mode': ['train'],
        'batch_size': [100],
        }
    main(**sweep_params)

# python tactile_baselines/her/supervised/pre_train.py
