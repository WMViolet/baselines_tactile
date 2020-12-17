import sys
import re
import os.path as osp
from envs import gym
from collections import defaultdict
import os
import json
import tensorflow as tf
import numpy as np
import time as timer
from experiment_utils.run_sweep import run_sweep
import tactile_baselines.her.experiment.config as config
from tactile_baselines.her.experiment.config import configure_her
from tactile_baselines.her.pointnet import get_features

from tactile_baselines.her.rollout import RolloutWorker
from tactile_baselines import logger

from importlib import import_module
from pdb import set_trace as st
import pickle
import dill as pickle

INSTANCE_TYPE = 'c4.xlarge'
EXP_NAME = 'supervised'

class ClassEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        if callable(o):
            return {'function': o.__name__}
        return json.JSONEncoder.default(self, o)

def nn(input, layers_sizes, reuse=None, flatten=False, name=""):
    """Creates a simple neural network
    """
    for i, size in enumerate(layers_sizes):
        activation = tf.nn.relu if i < len(layers_sizes) - 1 else None
        input = tf.layers.dense(inputs=input,
                                units=size,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                reuse=reuse,
                                name=name + '_' + str(i))
        if activation:
            input = activation(input)
    if flatten:
        assert layers_sizes[-1] == 1
        input = tf.reshape(input, [-1])
    return input


class FeatureNet:
    def __init__(self,
                 dims,
                 fixed_num_of_contact,
                 contact_dim,
                 sess,
                 process_type,
                 feature_dim,
                 feature_layer,
                 output = 'object'):
        self.o_tf = tf.placeholder(tf.float32, shape=[None, dims[0]])
        self.sess = sess
        self.dimo = dims[0]
        self.process_type = process_type
        self.feature_layer = feature_layer

        self.fixed_num_of_contact = fixed_num_of_contact
        self.contact_dim = contact_dim
        self.feature_dim = feature_dim
        self.feature_layer = feature_layer

        contact_info = self.o_tf[:, :self.contact_dim]
        self.contact_info = tf.reshape(contact_info, [-1, self.fixed_num_of_contact, self.contact_dim//self.fixed_num_of_contact])
        other_info = self.o_tf[:, self.contact_dim:]
        self.object_info = other_info[:, 48:]
        self.joint_info = other_info[:, :48]
        self.output = output

        if output == 'object':
            self.output_dim = self.object_info.shape[1]
        elif output == 'joint':
            self.output_dim = self.joint_info.shape[1]


        self.build_graph()



    def build_graph(self):
        if self.output == 'object':
            label = self.object_info
        elif self.output == 'joint':
            label = self.joint_info
        with tf.variable_scope('preprocess'):
            if self.process_type == 'pointnet':
                self.predictions, transform = get_features(self.contact_info, self.output_dim)
                classify_loss = tf.losses.mean_squared_error(predictions=self.predictions, labels=label, weights=0.5)
                K = transform.get_shape()[1].value
                mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
                mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
                mat_diff_loss = tf.nn.l2_loss(mat_diff)
                reg_weight=0.001
                self.total_loss = classify_loss + mat_diff_loss * reg_weight
            if self.process_type == 'max_pool':
                features = nn(self.contact_info, [self.feature_dim] * self.feature_layer + [self.output_dim])
                self.predictions = tf.reduce_max(tf.nn.relu(features), axis = 1)
                classify_loss = tf.losses.mean_squared_error(predictions=self.predictions, labels=label, weights=0.5)
                self.total_loss = classify_loss

        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3, name='optimizer')
        self.op = self.optimizer.minimize(loss=self.total_loss)
        self.pred_loss = classify_loss




    def train(self, data):
        feed_dict = {self.o_tf: data['o'].reshape((-1, self.dimo))}
        loss, _ = self.sess.run([self.total_loss, self.op], feed_dict=feed_dict)
        logger.logkv('train_classify_loss', loss)

    def test(self, data):
        feed_dict = {self.o_tf: data['o'].reshape((-1, self.dimo))}
        accuracy = self.sess.run([self.pred_loss], feed_dict=feed_dict)
        logger.logkv('test_pred_loss', accuracy[0])

    def predict_single(self, data):
        feed_dict = {self.o_tf: data.reshape((-1, self.dimo))}
        prediction = self.sess.run([self.predictions], feed_dict=feed_dict)[0]
        prediction = prediction.reshape((-1))
        return prediction




def split_data(paths, n):
    episode_size = paths.get_current_episode_size()
    logger.log("Collected episode size is ", episode_size)
    index = np.arange(episode_size)
    np.random.shuffle(index)
    train_index = index[:int(0.8 * episode_size)]
    test_index = index[int(0.8 * episode_size):]
    train_dict, test_dict = dict([]), dict([])
    for key in paths.buffers:
        data = paths.buffers[key]
        train_data, test_data = data[train_index], data[test_index]
        train_dict[key] = train_data
        test_dict[key] = test_data
    train_lst, test_lst = [], []
    for i in range(n):
        train, test = dict([]), dict([])
        for key in paths.buffers:
            train_data, test_data = train_dict[key], test_dict[key]
            train[key] = train_data[int(i/n)*100:(int(i/n) + 1)*100]
            test[key] = test_data[int(i/n)*100:(int(i/n) + 1)*100]
        train_lst.append(train)
        test_lst.append(test)

    return train_lst, test_lst



def main(**kwargs):
    exp_dir = os.getcwd() + '/data/' + EXP_NAME + '/' + str(kwargs['seed'])
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last')
    json.dump(kwargs, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = kwargs.get('gpu_frac', 0.95)
    sess = tf.Session(config=config)

    with sess.as_default() as sess:
        folder = './data/policy/' + kwargs['env']
        paths = pickle.load(open(folder + '/paths.pickle', 'rb'))
        niters = paths.get_current_episode_size() // 100
        train_data, test_data = split_data(paths, niters)

        dimo = train_data[0]['o'].shape[-1]

        dims = [dimo]
        env = gym.make(kwargs['env'], obs_type = kwargs['obs_type'], fixed_num_of_contact = kwargs['fixed_num_of_contact'])

        feature_net = FeatureNet(dims,
                                 fixed_num_of_contact = kwargs['fixed_num_of_contact'],
                                 contact_dim = env.contact_dim,
                                 sess = sess,
                                 output = kwargs['prediction'],
                                 process_type = kwargs['process_type'],
                                 feature_dim = kwargs['feature_dim'],
                                 feature_layer = kwargs['feature_layer'])

        sess.run(tf.global_variables_initializer())
        for i in range(niters):
            start = timer.time()
            feature_net.train(train_data[i])
            feature_net.test(test_data[i])
            logger.logkv("iter", i)
            logger.logkv("iter_time", timer.time() - start)
            logger.dumpkvs()
            if i == 0:
                sess.graph.finalize()




if __name__ == '__main__':
    sweep_params = {
        'alg': ['her'],
        'seed': [399856203240],
        'env': ['HandManipulateEgg-v0', 'HandManipulatePen-v0', 'HandManipulateBlock-v0'],
        'env': ['HandManipulateEgg-v0'],

        # Env Sampling
        'fixed_num_of_contact': [7],

        # Problem Conf
        'obs_type': ['contact', 'full_contact'],
        'obs_type': ['full_contact'],
        'process_type': ['max_pool', 'pointnet'],
        'process_type': ['pointnet'],
        'prediction': ['object'],
        'feature_dim': [128],
        'feature_layer': [0],
        }
    run_sweep(main, sweep_params, EXP_NAME, INSTANCE_TYPE)
