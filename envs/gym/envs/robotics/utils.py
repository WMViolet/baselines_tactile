import numpy as np
from pdb import set_trace as st

from envs.gym import error
try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

# should be between hand and object, check contact points between composite particles and particles
def get_valid_contacts(geom1s, geom2s, object_indices, positions, forces, hand_geoms = []):
    geoms = [[first, second] for first, second in zip(geom1s, geom2s)]
    filtered_geoms = [pair for pair in geoms if pair[0] in object_indices or pair[1] in object_indices]
    valid_indices = [idx for idx in range(len(geoms)) if geoms[idx][0] in object_indices or geoms[idx][1] in object_indices]
    if len(hand_geoms) != 0:
        filtered_geoms = [pair for pair in filtered_geoms if pair[0] in hand_geoms or pair[1] in hand_geoms]
        valid_indices = [idx for idx in valid_indices if geoms[idx][0] in hand_geoms or geoms[idx][1] in hand_geoms]
    geom1s, geom2s = [pair[0] for pair in filtered_geoms], [pair[1] for pair in filtered_geoms]
    num_of_contacts = len(geom1s)
    positions, forces = np.array(positions)[valid_indices], np.array(forces)[valid_indices]
    return geom1s, geom2s, num_of_contacts, positions, forces

def process_data(total_num_of_geoms,
                 num_of_contacts,
                 geom1s,
                 geom2s,
                 geom1_names,
                 geom2_names,
                 positions,
                 forces,
                 fixed_num_of_contact,
                 object_indices,
                 hand_geoms = [],
                 include_force = True,
                 include_geoms = True,
                 include_position = True):
    assert fixed_num_of_contact > 0
    dim = total_num_of_geoms * include_geoms + 3 * include_position + 6 * include_force
    if num_of_contacts == 0:
        return np.zeros((fixed_num_of_contact, dim)), 0
    geom1s, geom2s, num_of_contacts, positions, forces = get_valid_contacts(geom1s,
                                                         geom2s,
                                                         object_indices,
                                                         positions,
                                                         forces,
                                                         hand_geoms = hand_geoms)
    to_include = []
    if include_geoms:
        geom_array = np.zeros((num_of_contacts, total_num_of_geoms))
        for geom1, index in zip(geom1s, range(num_of_contacts)):
            geom_array[index][geom1] = 1
        for geom2, index in zip(geom2s, range(num_of_contacts)):
            geom_array[index][geom2] = 1
        to_include.append(geom_array)
    if include_position:
        to_include.append(positions)
    if include_force:
        to_include.append(forces)
    assert len(to_include) != 0
    concatenated = np.concatenate(to_include, axis=-1)

    if num_of_contacts < fixed_num_of_contact:
        empty = np.zeros((fixed_num_of_contact - num_of_contacts, dim))
        concatenated = np.concatenate((concatenated, empty), axis=0)
    else:
        concatenated = concatenated[:fixed_num_of_contact]
    return concatenated, num_of_contacts


def robot_get_obs(sim):
    """Returns all joint positions and velocities associated with
    a robot.
    """
    if sim.data.qpos is not None and sim.model.joint_names:
        names = [n for n in sim.model.joint_names if n.startswith('robot')]
        return (
            np.array([sim.data.get_joint_qpos(name) for name in names]),
            np.array([sim.data.get_joint_qvel(name) for name in names]),
        )
    return np.zeros(0), np.zeros(0)


def ctrl_set_action(sim, action):
    """For torque actuators it copies the action into mujoco ctrl field.
    For position actuators it sets the target relative to the current qpos.
    """
    if sim.model.nmocap > 0:
        _, action = np.split(action, (sim.model.nmocap * 7, ))
    if sim.data.ctrl is not None:
        for i in range(action.shape[0]):
            if sim.model.actuator_biastype[i] == 0:
                sim.data.ctrl[i] = action[i]
            else:
                idx = sim.model.jnt_qposadr[sim.model.actuator_trnid[i, 0]]
                sim.data.ctrl[i] = sim.data.qpos[idx] + action[i]


def mocap_set_action(sim, action):
    """The action controls the robot using mocaps. Specifically, bodies
    on the robot (for example the gripper wrist) is controlled with
    mocap bodies. In this case the action is the desired difference
    in position and orientation (quaternion), in world coordinates,
    of the of the target body. The mocap is positioned relative to
    the target body according to the delta, and the MuJoCo equality
    constraint optimizer tries to center the welded body on the mocap.
    """
    if sim.model.nmocap > 0:
        action, _ = np.split(action, (sim.model.nmocap * 7, ))
        action = action.reshape(sim.model.nmocap, 7)

        pos_delta = action[:, :3]
        quat_delta = action[:, 3:]

        reset_mocap2body_xpos(sim)
        sim.data.mocap_pos[:] = sim.data.mocap_pos + pos_delta
        sim.data.mocap_quat[:] = sim.data.mocap_quat + quat_delta


def reset_mocap_welds(sim):
    """Resets the mocap welds that we use for actuation.
    """
    if sim.model.nmocap > 0 and sim.model.eq_data is not None:
        for i in range(sim.model.eq_data.shape[0]):
            if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                sim.model.eq_data[i, :] = np.array(
                    [0., 0., 0., 1., 0., 0., 0.])
    sim.forward()


def reset_mocap2body_xpos(sim):
    """Resets the position and orientation of the mocap bodies to the same
    values as the bodies they're welded to.
    """

    if (sim.model.eq_type is None or
        sim.model.eq_obj1id is None or
        sim.model.eq_obj2id is None):
        return
    for eq_type, obj1_id, obj2_id in zip(sim.model.eq_type,
                                         sim.model.eq_obj1id,
                                         sim.model.eq_obj2id):
        if eq_type != mujoco_py.const.EQ_WELD:
            continue

        mocap_id = sim.model.body_mocapid[obj1_id]
        if mocap_id != -1:
            # obj1 is the mocap, obj2 is the welded body
            body_idx = obj2_id
        else:
            # obj2 is the mocap, obj1 is the welded body
            mocap_id = sim.model.body_mocapid[obj2_id]
            body_idx = obj1_id

        assert (mocap_id != -1)
        sim.data.mocap_pos[mocap_id][:] = sim.data.body_xpos[body_idx]
        sim.data.mocap_quat[mocap_id][:] = sim.data.body_xquat[body_idx]
