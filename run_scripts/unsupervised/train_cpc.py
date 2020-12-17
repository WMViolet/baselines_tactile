import numpy as np
import glob
import os

from envs import gym
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from utils.utils import *
from tactile_baselines import logger
from tactile_baselines.utils.utils import set_seed, ClassEncoder
from tactile_baselines.cpc.cpc_model import *
from tactile_baselines.cpc.data_util import *
import dill as pickle
import time

real_batch_size = 128

def get_neg_samples(start, end, obs, n = 50, b=1):
    """by default, return the same samples for the observations in the same batch"""
    other_rows = np.concatenate((np.arange(start), np.arange(end, obs.shape[0])), axis = -1)
    other_rows = obs[other_rows]
    _, _, epi_length, fixed_num_of_contact, contact_dim = other_rows.shape
    other_rows = other_rows.reshape((-1, fixed_num_of_contact, contact_dim))
    possibilities = other_rows.shape[0]
    if possibilities > n - 1:
        indices = np.random.permutation(possibilities)[:(n-1)*b]
        samples = other_rows[indices]
    else:
        print("ERROR: Need more points")
    return samples.reshape((b, n-1, fixed_num_of_contact, contact_dim))


def compute_cpc_loss(obs, obs_pos, obs_neg, encoder, trans, actions=None, type='dist'):
    """input shapes: obs: b x fixed_num_of_contact x contact_dim
                     obs_pos: b x fixed_num_of_contact x contact_dim
                     obs_neg: b x n x fixed_num_of_contact x contact_dim """
    z, z_pos = encoder(obs), encoder(obs_pos)  # b x z_dim

    if actions != None:
        actions = actions.float()
        z = torch.cat((z, actions), dim=1)
    z_next = trans(z)  # b x z_dim

    pos_log_density = (z_next * z_pos).sum(dim=1) # b x z_dim
    if type == 'cos':
        pos_log_density /= torch.norm(z_next, dim=1) * torch.norm(z_pos, dim=1)
    elif type == 'dist':
        pos_log_density = -((z_next ** 2).sum(1) - 2 * pos_log_density + (z_pos ** 2).sum(1))
    pos_log_density = pos_log_density.unsqueeze(1) # b x 1

    # obs_neg is b x n x 1 x 64 x 64
    bs, n_neg, _, _ = obs_neg.shape
    obs_neg = obs_neg.view(-1, *obs_neg.shape[2:]) # b * n x 1 x 64 x 64
    z_neg = encoder(obs_neg)  # b * n x z_dim

    z_next = z_next.unsqueeze(1)
    z_neg = z_neg.view(bs, n_neg, -1) # b x n x z_dim
    neg_log_density = torch.bmm(z_next, z_neg.permute(0, 2, 1).contiguous()).squeeze(1)  # b x n
    if type == 'cos':
        neg_log_density /= torch.norm(z_next, dim=2) * torch.norm(z_neg, dim=-1)
    elif type == 'dist':
        neg_log_density = -((z_next ** 2).sum(-1) - 2 * neg_log_density + (z_neg ** 2).sum(-1))

    loss = torch.cat((torch.zeros(bs, 1).cuda(), neg_log_density - pos_log_density), dim=1)  # b x n+1
    loss = torch.logsumexp(loss, dim=1).mean()
    return loss


def train_cpc(encoder, trans, optimizer, epoch, train_data, batch_size, n, k=1, include_action=True):
    """predict the next k steps. """
    global real_batch_size
    start = time.time()
    encoder.train()
    trans.train()

    train_losses = []
    train_loader = prep_data(train_data, batch_size, k, n)
    batch_num = train_loader[0].shape[0]
    batch_num = 100
    for idx in range(batch_num):
        obs, obs_pos = train_loader[0][idx], train_loader[2][idx]
        """add batch here, so that each sample in the batch get different neg samples. """
        obs, obs_pos = np.concatenate(obs), np.concatenate(obs_pos) #real_batch_size x fixed_num_of_contact x contact_dim
        obs, obs_pos = obs[:real_batch_size], obs_pos[:real_batch_size]
        obs_neg = get_neg_samples(idx*batch_size, (idx+1)*batch_size, train_loader[0], n = n, b = real_batch_size) # b x n x fixed_num_of_contact x contact_dim

        obs, obs_pos, obs_neg = torch.from_numpy(obs), torch.from_numpy(obs_pos), torch.from_numpy(obs_neg)
        obs, obs_pos = obs.cuda(), obs_pos.cuda(),  # b x 9 * contact_dim
        obs_neg = obs_neg.cuda() # (b x n) x 9 * contact_dim
        if include_action:
            actions = train_loader[1][idx]
            actions = np.concatenate(actions)
            actions = actions[:real_batch_size]
            actions = torch.from_numpy(actions)
            actions = actions.cuda()
            loss = compute_cpc_loss(obs, obs_pos, obs_neg, encoder, trans, actions=actions)
        else:
            loss = compute_cpc_loss(obs, obs_pos, obs_neg, encoder, trans, actions=None)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
    avg_loss = np.mean(train_losses[-50:])
    logger.logkv("cpc training loss", avg_loss)
    logger.logkv("cpc training time", time.time() - start)

def test_cpc(encoder, trans, epoch, test_data, batch_size, n, k=1, include_action = True):
    global real_batch_size
    start = time.time()
    encoder.eval()
    trans.eval()

    test_loss = 0
    test_loader = prep_data(test_data, batch_size, k, n)
    batch_num = test_loader[0].shape[0]
    batch_num = 20
    for idx in range(batch_num):
        obs, obs_pos = test_loader[0][idx], test_loader[2][idx]
        obs, obs_pos = np.concatenate(obs), np.concatenate(obs_pos) #real_batch_size x fixed_num_of_contact x contact_dim
        obs, obs_pos = obs[:real_batch_size], obs_pos[:real_batch_size]
        obs_neg = get_neg_samples(idx*batch_size, (idx+1)*batch_size, test_loader[0], n = n, b = real_batch_size) # b x n x fixed_num_of_contact x contact_dim
        obs, obs_pos, obs_neg = torch.from_numpy(obs), torch.from_numpy(obs_pos), torch.from_numpy(obs_neg)
        obs, obs_pos = obs.cuda(), obs_pos.cuda(),  # b x 9 * contact_dim
        obs_neg = obs_neg.cuda() # (b x n) x 9 * contact_dim
        if include_action:
            actions = test_loader[1][idx]
            actions = np.concatenate(actions)[:real_batch_size]
            actions = torch.from_numpy(actions)
            actions = actions.cuda()
            loss = compute_cpc_loss(obs, obs_pos, obs_neg, encoder, trans, actions=actions)
        else:
            loss = compute_cpc_loss(obs, obs_pos, obs_neg, encoder, trans, actions=None)
        test_loss += loss.item()
    avg_loss = test_loss/batch_num
    logger.logkv("cpc testing loss", avg_loss)
    logger.logkv("cpc testing time", time.time() - start)

def train_decoder(decoder, encoder, optim_dec, epoch, train_data, batch_size, include_action, n, k=1):
    global real_batch_size
    train_loader = prep_data(train_data, batch_size, k, n, decode = True)
    total_loss = 0
    batch_num = train_loader[0].shape[0]
    for idx in range(batch_num):
        obs = train_loader[0][idx]
        obs = np.concatenate(obs) #real_batch_size x fixed_num_of_contact x contact_dim
        obs = obs[:real_batch_size]
        obs = torch.from_numpy(obs)
        obs = obs.cuda()  # b x 9 * contact_dim
        object_info = train_loader[3][idx]
        object_info = torch.from_numpy(np.concatenate(object_info)[:real_batch_size]).cuda()
        recon = decoder(encoder(obs))
        loss = ((object_info-recon)**2).mean()
        optim_dec.zero_grad()
        loss.backward()
        optim_dec.step()
        total_loss += loss.item()
    logger.logkv("decoder training loss", total_loss)
    return total_loss/batch_num


def test_decoder(decoder, encoder, epoch, test_data, batch_size, include_action, n, k=1):
    global real_batch_size
    test_loader = prep_data(test_data, batch_size, k, n, decode = True)
    total_loss = 0
    batch_num = test_loader[0].shape[0]
    for idx in range(batch_num):
        obs = test_loader[0][idx]
        obs = np.concatenate(obs) #real_batch_size x fixed_num_of_contact x contact_dim
        obs = obs[:real_batch_size]
        obs = torch.from_numpy(obs)
        obs = obs.cuda()  # b x 9 * contact_dim
        recon = decoder(encoder(obs))
        object_info = test_loader[3][idx]
        object_info = torch.from_numpy(np.concatenate(object_info)[:real_batch_size]).cuda()
        loss = ((object_info-recon)**2).mean()
        total_loss += loss.item()
    logger.logkv("decoder testing loss", total_loss)
    return total_loss/batch_num


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
    exp_dir = os.getcwd() + '/cpc_model/' + kwargs['process_type'][0] + '/n200-8'
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last')
    json.dump(kwargs, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)

    obs, acts, fixed_num_of_contact = pickle.load(open('../untrained/HandManipulateEgg-v0/5seeds-dict.pickle', 'rb'))


    include_action = kwargs['include_action'][0]

    env = gym.make(kwargs['env'][0],
                   obs_type = kwargs['obs_type'][0],
                   fixed_num_of_contact = [fixed_num_of_contact, True])

    ngeoms = env.sim.model.ngeom
    obs, object_info = expand_data(obs, ngeoms, fixed_num_of_contact)
    next_obs = obs[:, 1:]
    obs = obs[:, :-1]
    N, L, _, contact_point_dim = obs.shape
    N, L, action_dim = acts.shape

    obs_dim = (fixed_num_of_contact, contact_point_dim)


    z_dim = 8
    lr = 1e-3
    epochs = 100
    batch_size = 2
    n = 200
    k = 1


    encoder = Encoder(z_dim, obs_dim[1], fixed_num_of_contact).cuda()
    if include_action:
        trans = Transition(z_dim, action_dim).cuda()
    else:
        trans = Transition(z_dim, 0).cuda()
    decoder = Decoder(z_dim, 3).cuda()


    optim_cpc = optim.Adam(list(encoder.parameters()) + list(trans.parameters()), lr=lr)
    optim_dec = optim.Adam(decoder.parameters(), lr=lr)
    train_data, test_data = split_data([obs, acts, next_obs])

    for epoch in range(epochs):
        train_cpc(encoder, trans, optim_cpc, epoch, train_data, batch_size, n, k, include_action)
        test_cpc(encoder, trans, epoch, test_data, batch_size, n, k, include_action)


        logger.logkv("epoch", epoch)
        logger.dumpkvs()


    train_data, test_data = split_data([obs, acts, next_obs, object_info])
    for epoch in range(100):
        train_decoder(decoder, encoder, optim_dec, epoch, train_data, batch_size, include_action, n, k=1)
        test_decoder(decoder, encoder, epoch, test_data, batch_size, include_action, n, k=1)
        logger.logkv("epoch", epoch)
        logger.dumpkvs()



if __name__ == '__main__':
    sweep_params = {
        'alg': ['her'],
        'seed': [5],
        'env': ['HandManipulateEgg-v0'],

        # Env Sampling
        'num_timesteps': [1e6],

        # Problem Confs
        'obs_type': ['full_contact'],
        'process_type': ['max_pool'],
        'input_label': ['positions'],
        'output_label': ['object_position'],
        'learning_rate': [1e-3],
        'mode': ['train'],
        'visualize_training_data': [False],
        'visualize_testing_data': [False],
        'batch_size': [100],
        'time_interval': [1],
        'include_action': [True],
        }
    main(**sweep_params)
    # python unsupervised/train_cpc.py
