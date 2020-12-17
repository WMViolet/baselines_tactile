import numpy as np
import glob
import os

from envs import gym
from utils.utils import *
from tactile_baselines import logger
from tactile_baselines.utils.utils import set_seed, ClassEncoder
from tactile_baselines.cpc.cpc_model_tf import *
from tactile_baselines.cpc.data_util import *
import dill as pickle
import time
from experiment_utils.run_sweep import run_sweep
import tensorflow as tf


INSTANCE_TYPE = 'c4.2xlarge'
EXP_NAME = 'cpc/hyper7'

real_batch_size = 128
def train_cpc(cpc, epoch, train_data, batch_size, n, k=1):
    """predict the next k steps. """
    start = time.time()
    train_losses = []
    train_loader = prep_data(train_data, batch_size, k, n)
    batch_num = train_loader[0].shape[0]
    batch_num = 100
    for idx in range(batch_num):
        obs, actions, obs_pos = train_loader[0][idx], train_loader[1][idx], train_loader[2][idx]
        obs_neg = get_neg_samples(obs, idx*batch_size, (idx+1)*batch_size, train_loader[0], n, cpc.type)
        obs, actions, obs_pos = np.concatenate(obs), np.concatenate(actions), np.concatenate(obs_pos) # b x fixed_num_of_contact * contact_dim
        obs_neg = obs_neg[:real_batch_size] # b x n x fixed_num_of_contact * contact_dim
        obs, actions, obs_pos = obs[:real_batch_size], actions[:real_batch_size], obs_pos[:real_batch_size]
        obs_neg = obs_neg.reshape((-1, *obs_neg.shape[-2:]))
        loss = cpc.train_encoder(obs, obs_pos, actions, obs_neg)
        train_losses.append(loss)

    losses = np.mean(train_losses[-50:])
    logger.logkv("cpc training loss", losses)
    logger.logkv("cpc training time", time.time() - start)

def test_cpc(cpc, epoch, test_data, batch_size, n, k=1):
    start = time.time()

    test_loss = 0
    test_loader = prep_data(test_data, batch_size, k, n)
    batch_num = test_loader[0].shape[0]
    batch_num = 20
    for idx in range(batch_num):
        obs, actions, obs_pos = test_loader[0][idx], test_loader[1][idx], test_loader[2][idx]
        obs_neg = get_neg_samples(obs, idx*batch_size, (idx+1)*batch_size, test_loader[0], n, cpc.type) # n x 9 * contact_dim
        obs, actions, obs_pos = np.concatenate(obs), np.concatenate(actions), np.concatenate(obs_pos)# b x 9 * contact_dim

        obs_neg = obs_neg[:real_batch_size] # n x fixed_num_of_contact * contact_dim
        obs, actions, obs_pos = obs[:real_batch_size], actions[:real_batch_size], obs_pos[:real_batch_size]
        obs_neg = obs_neg.reshape((-1, *obs_neg.shape[-2:]))

        loss = cpc.test_encoder(obs, obs_pos, actions, obs_neg)
        test_loss += loss

    test_loss /= batch_num
    logger.logkv("cpc testing loss", test_loss)
    logger.logkv("cpc testing time", time.time() - start)


def restore_cpc(cpc, epoch, test_data, batch_size, n, k=1, folder=''):
    test_loss = 0
    test_loader = prep_data(test_data, batch_size, k, n)
    batch_num = test_loader[0].shape[0]
    batch_num = 1
    idx = 0
    obs, actions, obs_pos = test_loader[0][idx], test_loader[1][idx], test_loader[2][idx]
    obs_neg = get_neg_samples(obs, idx*batch_size, (idx+1)*batch_size, test_loader[0], n, cpc.type)
    obs, actions, obs_pos = np.concatenate(obs), np.concatenate(actions), np.concatenate(obs_pos)

    obs, actions, obs_pos, obs_neg = obs[:real_batch_size], actions[:real_batch_size], obs_pos[:real_batch_size], obs_neg[:real_batch_size]
    obs_neg = obs_neg.reshape((-1, *obs_neg.shape[-2:]))

    test_loss = cpc.test(obs, obs_pos, actions, obs_neg)
    logger.logkv("cpc restored loss", test_loss)

    with open(folder + 'data.pickle', 'wb') as pickle_file:
        pickle.dump([obs, obs_pos, actions, obs_neg], pickle_file)

def train_decoder(decoder, epoch, train_data, batch_size, n, k=1):
    """predict the next k steps. """
    start = time.time()
    train_decoder_losses = []
    train_loader = prep_data(train_data, batch_size, k, n, decode = True)
    batch_num = train_loader[0].shape[0]
    batch_num = 300
    for idx in range(batch_num):
        obs = train_loader[0][idx]
        object_info = train_loader[3][idx]
        obs = np.concatenate(obs)[:real_batch_size]
        object_info = np.concatenate(object_info)[:real_batch_size]

        decoder_loss = decoder.train(obs, object_info)
        train_decoder_losses.append(decoder_loss)
    avg_loss = np.mean(train_decoder_losses[-50:])
    logger.logkv("decoder training loss", avg_loss)
    logger.logkv("decoder training time", time.time() - start)

def test_decoder(decoder, epoch, test_data, batch_size, n, k=1):
    start = time.time()
    decoder_test_loss = 0
    test_loader = prep_data(test_data, batch_size, k, n, decode = True)
    batch_num = test_loader[0].shape[0]
    batch_num = 50
    for idx in range(batch_num):
        obs = test_loader[0][idx]
        object_info = test_loader[3][idx]
        obs = np.concatenate(obs)[:real_batch_size] # b x 9 * contact_dim
        object_info = np.concatenate(object_info)[:real_batch_size]

        loss = decoder.test(obs, object_info)
        decoder_test_loss += loss
    decoder_test_loss /= batch_num
    logger.logkv("decoder testing loss", decoder_test_loss)
    logger.logkv("decoder testing time", time.time() - start)


def expand_data(paths, ngeoms, fixed_num_of_contact):
    N, R, _, _ = paths['geom2s'].shape
    geom2s = paths['geom2s'].reshape((N*R, fixed_num_of_contact)) #(N, 99, 9, 1) -> #(N, 99, 9)
    geom1s = paths['geom1s'].reshape((N*R, fixed_num_of_contact))
    geom_array = np.zeros((N*R, fixed_num_of_contact, ngeoms))
    indices1, indices2 = np.argwhere(geom1s!=-1), np.argwhere(geom2s!=-1)
    idx1, idx2 = (indices1[:, 0], indices1[:, 1]), (indices2[:, 0], indices2[:, 1])
    geom1, geom2 = geom1s[idx1].astype(int), geom2s[idx2].astype(int)
    idx1, idx2 = (idx1[0], idx1[1], geom1), (idx2[0], idx2[1], geom2)
    geom_array[idx1] = 1
    geom_array[idx2] = 1
    geom_array = geom_array.reshape((N, R, fixed_num_of_contact, ngeoms))
    concatenated = np.concatenate((geom_array, paths['positions'], paths['force']), axis=-1)
    object_info = paths['object_position'][:, :, :3]
    return concatenated, object_info


def main(**kwargs):
    z_dim = kwargs['z_dim']
    trans_mode = kwargs['trans_mode']
    epochs = kwargs['epochs']
    include_action = kwargs['include_action']
    label = kwargs['label']

    dataset = kwargs['data_path']
    feature_dims = kwargs['feature_dims']
    mode = kwargs['mode']
    n = kwargs['n']
    k = kwargs['k']
    encoder_lr = kwargs['encoder_lr']
    decoder_lr = kwargs['decoder_lr']
    decoder_feature_dims = kwargs['decoder_feature_dims']

    # if kwargs['data_path'] == '/home/vioichigo/try/tactile-baselines/dataset/sequence/HandManipulateEgg-v0/5seeds-dict.pickle':
    #     kwargs['dataset'] = 'trained_5seeds'
    # elif kwargs['data_path'] == '/home/vioichigo/try/tactile-baselines/dataset/untrained/HandManipulateEgg-v0/5seeds-dict.pickle':
    #     kwargs['dataset'] = 'untrained_5seeds'
    # elif kwargs['data_path'] == '/home/vioichigo/try/tactile-baselines/dataset/HandManipulateEgg-v09-dict.pickle':
    #     kwargs['dataset'] = 'trained_1seed'
    exp_dir = os.getcwd() + '/data/' + EXP_NAME
     # + '/' + str(time.time())
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last')
    json.dump(kwargs, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = kwargs.get('gpu_frac', 0.95)
    sess = tf.Session(config=config)

    obs, acts, fixed_num_of_contact = pickle.load(open(dataset, 'rb'))

    env = gym.make(kwargs['env'],
                   obs_type = kwargs['obs_type'],
                   fixed_num_of_contact = [fixed_num_of_contact, True])

    ngeoms = env.sim.model.ngeom
    obs, object_info = expand_data(obs, ngeoms, fixed_num_of_contact)
    next_obs = obs[:, 1:]
    obs = obs[:, :-1]
    N, L, _, contact_point_dim = obs.shape
    N, L, action_dim = acts.shape

    obs_dim = (fixed_num_of_contact, contact_point_dim)
    train_data, test_data = split_data([obs, acts, next_obs, object_info])

    batch_size = 2

    if mode == 'restore':
        saver = tf.train.import_meta_graph(exp_dir + '-999.meta')
        saver.restore(sess, tf.train.latest_checkpoint(exp_dir))
        graph = tf.get_default_graph()

    with sess.as_default() as sess:
        encoder = Encoder(z_dim,
                          fixed_num_of_contact,
                          contact_point_dim,
                          feature_dims)
        trans = Transition(z_dim, action_dim, mode = trans_mode)
        cpc = CPC(sess,
                  encoder,
                  trans,
                  encoder_lr,
                  fixed_num_of_contact,
                  contact_point_dim,
                  action_dim,
                  include_action = include_action,
                  type = 1*(label=='cpc1') + 2*(label=='cpc2'),
                  n_neg = n)

        cpc_epochs, decoder_epochs = epochs

        if mode == 'train':
            sess.run(tf.global_variables_initializer())
            logger.log("training started")
            for epoch in range(cpc_epochs):
                train_cpc(cpc, epoch, train_data, batch_size, n, k)
                test_cpc(cpc, epoch, test_data, batch_size, n, k)

                logger.logkv("epoch", epoch)
                logger.dumpkvs()
            cpc.save_model(exp_dir, 999)

            """decoder"""
            logger.log("Done with cpc training.")

            decoder = Decoder(cpc,
                              sess,
                              z_dim,
                              decoder_feature_dims,
                              fixed_num_of_contact,
                              contact_point_dim,
                              decoder_lr)
            uninit_vars = [var for var in tf.global_variables() if not sess.run(tf.is_variable_initialized(var))]
            sess.run(tf.variables_initializer(uninit_vars))
            for epoch in range(decoder_epochs):
                train_decoder(decoder, epoch, train_data, batch_size, n, k)
                test_decoder(decoder, epoch, test_data, batch_size, n, k)

                logger.logkv("epoch", (epoch + cpc_epochs))
                logger.dumpkvs()

        if mode == 'restore':
            cpc.restore_encoder()
            for i in range(1):
                logger.logkv("iter", i)
                with open(exp_dir + 'params.pickle', 'wb') as pickle_file:
                    pickle.dump([z_dim,
                                 fixed_num_of_contact,
                                 contact_point_dim,
                                 action_dim,
                                 encoder_lr,
                                 feature_dims,
                                 trans_mode], pickle_file)
                restore_cpc(cpc, 0, test_data, batch_size, n, k, folder = exp_dir)
                logger.dumpkvs()
        tf.reset_default_graph()
        print("graph reset successfully")



if __name__ == '__main__':
    sweep_params = {
        'alg': ['her'],
        'seed': [1, 2],
        'env': ['HandManipulateEgg-v0'],

        # Env Sampling
        'num_timesteps': [1e6],

        # Problem Confs
        'obs_type': ['object_loc+rot+geom+contact_loc+force+other'],
        'mode': ['train'],
        'encoder_lr': [1e-3],
        'decoder_lr': [1e-3],
        'decoder_feature_dims': [[32, 32]],
        'feature_dims': [[64, 64], [8, 8]],
        'trans_mode': ['MLP'], #MLP and linear
        'z_dim': [32, 8],
        'epochs': [[200, 200]],
        'n': [100, 200],
        'k': [1],
        'include_action': [True, False],
        'label': ['cpc1', 'cpc2'],
        # 'data_path': ['../dataset/sequence/HandManipulateEgg-v0/5seeds-dict.pickle', '../dataset/untrained/HandManipulateEgg-v0/5seeds-dict.pickle', '../dataset/HandManipulateEgg-v09-dict.pickle'],
        'data_path': ['/home/vioichigo/try/tactile-baselines/dataset/sequence/HandManipulateEgg-v0/2seeds-dict.pickle']
        }
    run_sweep(main, sweep_params, EXP_NAME, INSTANCE_TYPE)
    # main(**sweep_params)
    # python run_scripts/unsupervised/train_cpc_tf.py
