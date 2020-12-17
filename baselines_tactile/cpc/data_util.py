import numpy as np
from pdb import set_trace as st


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

def split_data(data_list):
    n = data_list[0].shape[0]
    indices = np.random.permutation(n)
    train_data = []
    test_data = []
    split = int(0.8 * n)
    for data in data_list:
        data = data[indices]
        train_data.append(data[:split])
        test_data.append(data[split:])
    return train_data, test_data
    # obs[:split], obs[split:], acts[:split], acts[split:], next_obs[:split], next_obs[split:]

def prep_data(train_data, batch_size, k=1, n = 50, decode = False):
    full_obs, full_acts, full_next_obs, object_info = train_data
    # if decode:
    #     full_obs, full_acts, full_next_obs, object_info = train_data
    # else:
    #     full_obs, full_acts, full_next_obs = train_data
    num = full_obs.shape[0]
    if num >= batch_size:
        indices = np.random.permutation(num)[:batch_size * (num//batch_size)]
        full_obs, full_acts, full_next_obs = full_obs[indices], full_acts[indices], full_next_obs[indices]
        if decode:
            object_info = object_info[indices]
            object_info = object_info[:, :-1]
    else:
        print("WARNING: batch_size too large. ")
        batch_size = num
    # next observations should be removed...
    # choose positive samples
    assert k == 1 # only for now
    pos = full_next_obs #(8050, 99, 9, 58)

    _, epi_length, fixed_num_of_contact, contact_dim = full_next_obs.shape
    _, _, act_dim = full_acts.shape
    full_obs = full_obs.reshape((-1, batch_size, epi_length, fixed_num_of_contact, contact_dim)) #(252, 32, 99, 9, 58)
    full_acts = full_acts.reshape((-1, batch_size, epi_length, act_dim)) #(252, 32, 99, 20)
    pos = pos.reshape((-1, batch_size, epi_length, fixed_num_of_contact, contact_dim))
    if decode:
        object_info = object_info.reshape((-1, batch_size, epi_length, 3))
        return full_obs, full_acts, pos, object_info
    else:
        return full_obs, full_acts, pos

def get_neg_samples(batch, start, end, obs, n = 50, type = 2, alpha = 0.2):
    """EDITED: 0.8 neg samples from other trajectories
               0.2 neg samples from the same trajectories"""
    epi_num, epi_length, fixed_num_of_contact, contact_dim = batch.shape
    bs = epi_num*epi_length
    from_same_traj = int(alpha * (n-1))
    from_other_traj = ((n-1) - from_same_traj)
    if type == 2:
        from_same_traj *= bs
        from_other_traj *= bs
    elif type == 1:
        from_other_traj = n-1
    # a more efficient way of selecting from other trajectories
    other_rows = np.concatenate((np.arange(start), np.arange(end, obs.shape[0])), axis = -1) #9998
    other_rows_indices = np.random.randint(other_rows.shape[0], size=from_other_traj)
    other_rows = other_rows[other_rows_indices]
    other_epis = np.random.randint(obs.shape[1], size=from_other_traj)
    other_columns = np.random.randint(obs.shape[2], size=from_other_traj)
    other_samples = obs[other_rows, other_epis, other_columns] # from_other_traj x fixed_num x contact_dim
    # select from the same trajectories
    # only i and i+1 are not qualified, need bs x 20
    # own_indices = np.concatenate((np.arange(epi_length), np.arange(epi_length)), axis = -1)
    if type == 2:
        other_samples = other_samples.reshape((bs, -1, fixed_num_of_contact, contact_dim))
        from_same_traj = from_same_traj//epi_num
        same_epis = np.repeat(np.arange(epi_num), from_same_traj, axis = 0)
        # remove one possibility for easier computation
        qualified = np.array([np.concatenate((np.arange(i), np.arange(i+2, epi_length+1)), axis = -1)[:-1] for i in range(epi_length)])
        same_indices = np.random.randint(qualified.shape[1], size=(from_same_traj)) #99x20
        same_column1 = qualified[np.repeat(np.arange(epi_length), from_same_traj//epi_length, axis = 0), same_indices]
        same_indices = np.random.randint(qualified.shape[1], size=(from_same_traj)) #99x20
        same_column2 = qualified[np.repeat(np.arange(epi_length), from_same_traj//epi_length, axis = 0), same_indices]
        same_columns = np.concatenate((same_column1, same_column2), axis = 0)
        same_samples = batch[same_epis.astype(int), same_columns]
        # if type == 2:
        same_samples = same_samples.reshape((bs, -1, fixed_num_of_contact, contact_dim))
        samples = np.concatenate((other_samples, same_samples), axis = 1) #BxNxFxC
    #     samples = np.concatenate((other_samples, same_samples), axis = 0) #NxFxC
    elif type == 1:
        samples = other_samples
    return samples
