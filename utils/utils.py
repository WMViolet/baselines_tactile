import json
import tensorflow as tf
from pdb import set_trace as st
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

def print_episode(episode):
    epi_len, fixed_num_of_contact, contact_dim = episode.shape
    for x in episode:
        for i in range(fixed_num_of_contact):
            print("********************")
            if len(np.argwhere(x[i]==1)) != 0:
                print(np.argwhere(x[i]==1))
                print(x[i][-9:])
            print("********************")

def data_filter(data, fixed_num_of_contact, batch_size, min_num_points = 0):
    """remove the data where the egg fails down. """
    assert 'geom1s' in data
    # indices = (np.sum(data['geom1s'], axis = 1).reshape((-1)) != -fixed_num_of_contact)
    if min_num_points > 0:
        indices = (data['geom1s'][:, min_num_points-1, 0] > -1)
    else:
        indices = np.array([True]*(data['geom1s'].shape[0]))
    valid_data = {}
    data_num = np.sum(indices)
    processed_num = (data_num//batch_size) * batch_size
    for key in data:
        # if key not in ['geom1s', 'geom2s', 'force']:
        if key not in ['object_vel', 'joint_position', 'joint_vel']:
            valid_data[key] = data[key][indices][:processed_num]

    valid_data['object_position'] = valid_data['object_position'][:, :-4]
    return valid_data

def split_data(paths, niter):
    episode_size = paths['positions'].shape[0]
    index = np.arange(episode_size)
    np.random.shuffle(index)
    train_index = index[:int(0.8 * episode_size)]
    test_index = index[int(0.8 * episode_size):]
    train_dict, test_dict = {}, {}
    for key in paths.keys():
        data = paths[key]
        train_data, test_data = data[train_index], data[test_index]
        if len(train_data.shape) == 2:
            train_dict[key] = train_data.reshape((niter, -1, train_data.shape[1]))
            test_dict[key] = test_data.reshape((niter, -1, train_data.shape[1]))
        elif len(train_data.shape) == 3:
            train_dict[key] = train_data.reshape((niter, -1, train_data.shape[1], train_data.shape[2]))
            test_dict[key] = test_data.reshape((niter, -1, train_data.shape[1], train_data.shape[2]))

    return train_dict, test_dict



def expand_data(paths, ngeoms, fixed_num_of_contact):
    geom2s = paths['geom2s'].reshape((-1, fixed_num_of_contact))
    geom1s = paths['geom1s'].reshape((-1, fixed_num_of_contact))
    ndata = geom2s.shape[0]
    geom_array = np.zeros((ndata, fixed_num_of_contact, ngeoms))
    indices1, indices2 = np.argwhere(geom1s!=-1), np.argwhere(geom2s!=-1)
    idx1, idx2 = (indices1[:, 0], indices1[:, 1]), (indices2[:, 0], indices2[:, 1])
    geom1, geom2 = geom1s[idx1].astype(int), geom2s[idx2].astype(int)
    idx1, idx2 = (idx1[0], idx1[1], geom1), (idx2[0], idx2[1], geom2)
    geom_array[idx1] = 1
    geom_array[idx2] = 1
    concatenated = np.concatenate((geom_array, paths['positions'], paths['force']), axis=-1)
    paths['contacts'] = concatenated
    paths['geoms'] = geom_array
    return paths




def visualize_data(paths, env, fixed_num_of_contact, feature_net, mode, input_label, predict = 'both', time_interval = 1, site_mode = 'sequential'):
    data_num = paths['positions'].shape[0]
    print("number of data: ", data_num)
    print(predict)
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
        time.sleep(time_interval)

        dim = 3
        if site_mode == 'sequential':
            for contact_idx in range(fixed_num_of_contact):
                if sum(positions[contact_idx][-dim:] == np.zeros(dim)) != dim:
                    site_name = 'contact{}'.format(contact_idx+1)
                    site_id = env.sim.model.site_name2id(site_name)
                    env.sim.model.site_pos[site_id] = positions[contact_idx]
                    env.sim.forward()
                    env.render()
                    time.sleep(time_interval)
        elif site_mode == 'concurrent':
            for contact_idx in range(fixed_num_of_contact):
                if sum(positions[contact_idx][-dim:] == np.zeros(dim)) != dim:
                    site_name = 'contact{}'.format(contact_idx+1)
                    site_id = env.sim.model.site_name2id(site_name)
                    env.sim.model.site_pos[site_id] = positions[contact_idx]
                    env.sim.forward()
            env.render()
            time.sleep(time_interval)

        if mode == 'train':
            prediction = feature_net.predict_single(inputs.reshape((1, fixed_num_of_contact, -1)))
        elif mode == 'restore':
            prediction = feature_net.restore_predict_single(inputs.reshape((1, fixed_num_of_contact, -1)))

        env.sim.data.set_joint_qpos('object:joint', prediction)
        env.sim.forward()
        env.render()
        time.sleep(time_interval)
        for contact_idx in range(fixed_num_of_contact):
            site_name = 'contact{}'.format(contact_idx+1)
            site_id = env.sim.model.site_name2id(site_name)
            env.sim.model.site_pos[site_id] = np.array([1, 0.9, 0.25])
            env.sim.forward()
        env.sim.data.set_joint_qpos('object:joint', 2*np.ones(7))
        env.render()
        time.sleep(time_interval)
    env.close()


def geom_names_and_indices(env):
    sim = env.sim
    ngeom = sim.model.ngeom
    for idx in range(ngeom):
        print(idx, sim.model.geom_id2name(idx))
