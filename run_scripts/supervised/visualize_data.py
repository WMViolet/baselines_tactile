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
from tactile_baselines.her.pointnet import get_features

from tactile_baselines import logger
from tactile_baselines.utils.utils import set_seed, ClassEncoder
from pdb import set_trace as st
from utils.utils import expand_data, split_data


def visualize_data(paths, env, fixed_num_of_contact, feature_net, mode, input_label):
    data_num = paths['force'].shape[0]
    print("number of data: ", data_num)
    for idx in range(data_num):
        object_position = paths['original_object_position'][idx]
        object_vel = paths['object_vel'][idx]
        joint_position = paths['joint_position'][idx]
        joint_vel = paths['joint_vel'][idx]
        inputs = paths[input_label][idx]
        positions = paths['positions'][idx]


        env.sim.data.set_joint_qpos('object:joint', object_position)
        env.sim.data.set_joint_qvel('object:joint', object_vel)

        for idx in range(len(env.sim.model.joint_names)):
            name = env.sim.model.joint_names[idx]
            if name.startswith('robot'):
                env.sim.data.set_joint_qpos(name, joint_position[idx])
                env.sim.data.set_joint_qvel(name, joint_vel[idx])
        env.sim.forward()
        env.render()
        time.sleep(1)

        dim = 3
        for contact_idx in range(fixed_num_of_contact):
            if sum(positions[contact_idx][-dim:] == np.zeros(dim)) != dim:
                site_name = 'contact{}'.format(contact_idx+1)
                site_id = env.sim.model.site_name2id(site_name)
                env.sim.model.site_pos[site_id] = positions[contact_idx]
                env.sim.forward()
                env.render()
                time.sleep(1)


        for contact_idx in range(fixed_num_of_contact):
            site_name = 'contact{}'.format(contact_idx+1)
            site_id = env.sim.model.site_name2id(site_name)
            env.sim.model.site_pos[site_id] = np.array([1, 0.9, 0.25])
            env.sim.forward()
        env.sim.data.set_joint_qpos('object:joint', 2*np.ones(7))
        env.render()
        time.sleep(1)
    env.close()



def main(**kwargs):
    import dill as pickle
    from datetime import datetime
    exp_dir = os.getcwd() + '/data/feature_net/' + kwargs['input_label'][0] + kwargs['output_label'][0] + '/'
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last')
    json.dump(kwargs, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = kwargs.get('gpu_frac', 0.95)
    sess = tf.Session(config=config)

    mode = kwargs['mode'][0]
    visualize_training_data = kwargs['visualize_training_data'][0]
    visualize_testing_data = kwargs['visualize_testing_data'][0]
    visualize_new_data = kwargs['visualize_new_data'][0]

    if mode == 'restore':
        saver = tf.train.import_meta_graph(exp_dir + '-999.meta')
        saver.restore(sess, tf.train.latest_checkpoint(exp_dir))
        graph = tf.get_default_graph()

    with sess.as_default() as sess:

        # folder = './data/policy/' + kwargs['env'][0]
        # buffer, fixed_num_of_contact = pickle.load(open('../saved/HandManipulateEgg-v0-fix9.pickle', 'rb'))

        buffer = {}
        name = 's1'
        paths, fixed_num_of_contact = pickle.load(open('../saved/soft/' + name + '80-dict.pickle', 'rb'))
        for key in paths:
            buffer[key] = paths[key]

        for name in ['s2', 's4', 's5', 's6', 'soft3']:
            paths, fixed_num_of_contact = pickle.load(open('../saved/soft/' + name + '80-dict.pickle', 'rb'))
            for key in paths:
                buffer[key] = np.concatenate([buffer[key], paths[key]], axis = 0)


        env = gym.make(kwargs['env'][0],
                       obs_type = kwargs['obs_type'][0],
                       fixed_num_of_contact = fixed_num_of_contact)

        for key in buffer:
            buffer[key] = buffer[key][:int(1e6)]


        niters = buffer['positions'].shape[0] // 100
        print("total iteration: ", niters)


        ngeoms = env.sim.model.ngeom
        input_label = kwargs['input_label'][0]
        output_label = kwargs['output_label'][0]
        start = time.time()
        # paths = expand_data(buffer, ngeoms, fixed_num_of_contact, input_label, output_label)
        # print("expand data:", time.time() - start)
        paths = buffer

        start = time.time()
        train_data, test_data, vis_data, vis_data_test = split_data(paths, niters)
        print("split data:", time.time() - start)

        train_data['object_position'] = train_data['object_position'][:, :, :3]
        vis_data['original_object_position'] = vis_data['object_position']
        vis_data_test['original_object_position'] = vis_data_test['object_position']
        test_data['object_position'] = test_data['object_position'][:, :, :3]

        labels_to_dims = {}
        labels_to_dims['contacts'] = 3+6+ngeoms
        labels_to_dims['positions'] = 3
        # labels_to_dims['object_position'] = 7
        labels_to_dims['object_position'] = 3
        labels_to_dims['joint_position'] = 24
        labels_to_dims['object_vel'] = 6
        labels_to_dims['joint_vel'] = 24
        labels_to_dims['geoms'] = ngeoms



        dims = (labels_to_dims[input_label], labels_to_dims[output_label])
        print("preparation done")



        num_episodes = 1
        horizon = 100
        if visualize_training_data:
            visualize_data(vis_data, env, fixed_num_of_contact, feature_net, mode, input_label)
        if visualize_testing_data:
            visualize_data(vis_data_test, env, fixed_num_of_contact, feature_net, mode, input_label)




if __name__ == '__main__':
    sweep_params = {
        'alg': ['her'],
        'seed': [1],
        'env': ['SoftHandManipulateEgg-v0'],

        # Env Sampling
        'num_timesteps': [1e6],
        'buffer_size': [1e6],

        # Problem Conf
        'obs_type': ['full_contact'],
        'process_type': ['max_pool'],
        'input_label': ['positions'],
        'output_label': ['object_position'],
        'feature_layers': [[64, 64]],
        'output_layers': [[64, 64]],
        'mode': ['train'],
        'visualize_training_data': [False],
        'visualize_testing_data': [False],
        'visualize_new_data': [False]
        }
    main(**sweep_params)
