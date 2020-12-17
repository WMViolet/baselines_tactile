import tensorflow as tf
import torch
import torch.nn as nn
import numpy as np
from pdb import set_trace as st
from torch.autograd import Variable
from tactile_baselines.cpc.data_util import *
import dill as pickle
import torch.optim as optim
from unsupervised.train_cpc import *

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

batch_size = 1
k = 1
n = 50
z_dim = 8
fixed_num_of_contact = 9
include_action = True

train_obs, train_acts, train_next_obs = pickle.load(open('../sequence/debug_data.pickle', 'rb')) #99, 9, dim
_, _, _, contact_point_dim = train_obs.shape
_, _, action_dim = train_acts.shape
lr = 2e-4

from tactile_baselines.cpc.cpc_model_tf import *
encoder = Encoder(z_dim, fixed_num_of_contact, contact_point_dim)
trans = Transition(z_dim, action_dim)
optim_cpc = tf.train.AdamOptimizer(learning_rate=lr)
cpc = CPC(sess,
          encoder,
          trans,
          optim_cpc,
          fixed_num_of_contact,
          contact_point_dim,
          action_dim,
          n-1,
          include_action)

from tactile_baselines.cpc.cpc_model import *
encoder = Encoder(z_dim, contact_point_dim, fixed_num_of_contact).cuda()
trans = Transition(z_dim, action_dim).cuda()
optim_cpc = optim.Adam(list(encoder.parameters()) + list(trans.parameters()), lr=lr)

with sess.as_default() as sess:
    sess.run(tf.global_variables_initializer())

    train_loader = prep_data([train_obs, train_acts, train_next_obs], batch_size, k, n)
    batch_num = train_loader[0].shape[0]
    for idx in range(batch_num):
        obs, actions, obs_pos = train_loader[0][idx], train_loader[1][idx], train_loader[2][idx]
        obs_neg = get_neg_samples(idx*batch_size, (idx+1)*batch_size, train_loader[0])
        obs, actions, obs_pos = np.concatenate(obs), np.concatenate(actions), np.concatenate(obs_pos)
        obs, obs_pos, actions = obs, obs_pos, actions # b x 9 * contact_dim
        obs_neg = obs_neg # n x 9 * contact_dim

        """tensorflow"""

        tf_loss = cpc.train(obs, obs_pos, actions, obs_neg)

        """pytorch"""
        obs = torch.from_numpy(obs).cuda()
        obs_pos = torch.from_numpy(obs_pos).cuda()
        obs_neg = torch.from_numpy(obs_neg).cuda()
        actions = torch.from_numpy(actions).cuda()
        torch_loss = compute_cpc_loss(obs, obs_pos, obs_neg, encoder, trans, actions=actions)
        print("pytorch loss: ", torch_loss.item())
        print("tensorflow loss: ", tf_loss)
        st()


# avg_loss = np.mean([loss][-50:])
# losses += avg_loss
# losses = losses/(batch_num * 99)
