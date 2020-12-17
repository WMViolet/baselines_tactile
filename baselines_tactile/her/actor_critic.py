import tensorflow as tf
from pdb import set_trace as st
from tactile_baselines.her.util import store_args, nn, nn_max_pool
from tactile_baselines.her.supervised.model import supervised_nn_max_pool
from tactile_baselines.cpc.cpc_model_tf import to_latent, predict
import dill as pickle
# from tensorflow.contrib.staging import StagingArea


class ActorCritic:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u,
                 o_stats, g_stats, hidden, layers,
                 process_type, fixed_num_of_contact = 0,
                 contact_dim = 0,
                 **kwargs):

        """The actor-critic network and related training code.
        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (tactile_baselines.her.Normalizer): normalizer for observations
            g_stats (tactile_baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """

        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']
        # only normalize object information

        contact_point_dim = contact_dim//fixed_num_of_contact

        g = self.g_stats.normalize(self.g_tf)

        if self.use_contact:
            contact_info = tf.reshape(self.o_tf[:, :contact_dim], (-1, fixed_num_of_contact, contact_point_dim))
            object_info = self.o_tf[:, contact_dim:]

            if self.pre_train == 'supervised':
                with tf.variable_scope('pi/process'):
                    pos_layers = pickle.load(open(kwargs['feature_net_path'] + 'layers.pickle', 'rb'))
                    contact_pos = contact_info[:, :, -9:-6]
                    self.features = supervised_nn_max_pool(contact_pos, pos_layers[0], pos_layers[1] + [3], name = 'predicted_pos', reuse=tf.AUTO_REUSE)

                    # predicted_o = tf.concat([object_info[:, :-7], self.features, object_info[:, -4:]], axis = -1)

                    """start"""
                    geoms = contact_info[:, :, :-9]
                    geom_info = tf.reduce_max(geoms, axis = 1)

                    o = tf.concat([self.o_tf[:, :contact_dim], object_info[:, :-7], self.features, object_info[:, -4:]], axis = -1)
                    o = self.o_stats.normalize(o)

                    contact_info = tf.reshape(o[:, :contact_dim], (-1, fixed_num_of_contact, contact_point_dim))
                    object_info = o[:, contact_dim:]
                    force_info = contact_info[:, 0, -6:]
                    features = tf.concat([geom_info, force_info], axis = -1)
                    o = tf.concat([features, object_info], axis = -1)

                    # o = self.o_stats.normalize(predicted_o)
            elif self.pre_train == 'cpc':
                with tf.variable_scope('pi/process'):
                    z_dim, fixed_num_of_contact, contact_point_dim, action_dim, encoder_lr, feature_dims, trans_mode, label, include_action = pickle.load(open(self.feature_net_path + 'params.pickle', 'rb'))
                    self.obs_neg = self.cpc_inputs_tf['obs_neg']
                    self.obs_pos = self.cpc_inputs_tf['obs_pos']
                    self.z_dim, self.n_neg, self.type = z_dim, 100, 1 * (label == 'cpc1') + 2 * (label == 'cpc2')

                    # if self.process_type == 'none':
                    #     contact_info = tf.reshape(contact_info, [-1, contact_dim])
                    self.features = to_latent(x = contact_info, z_dim = z_dim, layers_sizes = feature_dims, name='new_cpc', process_type = self.process_type)
                    self.pos_features = to_latent(x = self.obs_pos, z_dim = z_dim, layers_sizes = feature_dims, name='new_cpc', process_type = self.process_type)
                    self.neg_features = to_latent(x = self.obs_neg, z_dim = z_dim, layers_sizes = feature_dims, name='new_cpc', process_type = self.process_type)
                    if include_action:
                        z = tf.concat((self.features, self.u_tf), axis=1)
                    else:
                        z = self.features
                    self.next = predict(z, z_dim = z_dim, name = 'new_cpc/trans_graph')  # b x z_dim

                    features = self.features
                    features = tf.tanh(features)
                    features = tf.clip_by_value(features, -self.clip_obs, self.clip_obs)
                    features = self.feature_stats.normalize(features)
                    object_info = self.o_stats.normalize(object_info)
                    o = tf.concat([features, object_info], axis = -1)
            elif self.pre_train == 'curl':
                with tf.variable_scope('pi/curl'):
                    o = self.o_stats.normalize(self.o_tf)
                    contact_info = tf.reshape(o[:, :contact_dim], (-1, fixed_num_of_contact, contact_point_dim))
                    # contact_info = o[:, :contact_dim]
                    object_info = o[:, contact_dim:]
                    self.z_dim = 32
                    # self.features = tf.tanh(tf.reduce_max(nn(contact_info, [self.z_dim] * 2), axis = 1))
                    self.features = tf.tanh(nn(contact_info, [self.z_dim] * 2))
                    features = self.features
                    # features = tf.clip_by_value(features, -self.clip_obs, self.clip_obs)
                    # features = self.feature_stats.normalize(features)
                    o = tf.concat([features, object_info], axis = -1)
            else:
                # geoms = contact_info[:, :, :-9]
                # geom_info = tf.reduce_max(geoms, axis = 1)
                o = self.o_stats.normalize(self.o_tf)
                object_info = o[:, contact_dim:]
                contact_info = tf.reshape(o[:, :contact_dim], (-1, fixed_num_of_contact, contact_point_dim))
                # contact_info = tf.concat([geoms, contact_info[:, :, -9:]], axis = -1)
                # force_info = contact_info[:, 0, -6:]
                # pos = contact_info[:, :, -9:-6]
                with tf.variable_scope('pi/features'):
                    features = nn_max_pool(contact_info, [128] * 2)
                    # info = nn(contact_info, [128] * 2)
                    # info = tf.reshape(info, [-1, np.prod(info.shape[1:])])
                #
                # st()
                # pos = tf.reshape(contact_info[:, :, -9:-6], [-1, fixed_num_of_contact * 3])
                # features = tf.concat([geom_info, pos, force_info], axis = -1)
                o = tf.concat([features, object_info], axis = -1)
                # o = object_info
        else:
            o = self.o_stats.normalize(self.o_tf)




        input_pi = tf.concat(axis=1, values=[o, g])  # for actor

        # Networks.
        with tf.variable_scope('pi'):
            self.pi_tf = self.max_u * tf.tanh(nn(
                input_pi, [self.hidden] * self.layers + [self.dimu]))

        with tf.variable_scope('Q'):
            # for policy training
            input_Q = tf.concat(axis=1, values=[o, g, self.pi_tf / self.max_u])
            self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])
            # for critic training
            input_Q = tf.concat(axis=1, values=[o, g, self.u_tf / self.max_u])
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)
