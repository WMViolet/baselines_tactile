import tensorflow as tf
from pdb import set_trace as st
from tactile_baselines.her.util import store_args, nn, nn_max_pool
from tactile_baselines.her.supervised.model import supervised_nn_max_pool
from tactile_baselines.cpc.cpc_model_tf import to_latent
import dill as pickle



class SoftActorCritic:
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
        self.o_2_tf = inputs_tf['o_2']
        self.g_2_tf = inputs_tf['g_2']
        self.u_tf = inputs_tf['u']
        # only normalize object information

        contact_point_dim = contact_dim//fixed_num_of_contact

        g = self.g_stats.normalize(self.g_tf)

        if self.use_contact:
            contact_info = tf.reshape(self.o_tf[:, :contact_dim], (-1, fixed_num_of_contact, contact_point_dim))
            object_info = self.o_tf[:, contact_dim:]

            if self.pre_train == 'feature_net':
                with tf.variable_scope('pi/process'):
                    pos_layers = pickle.load(open(kwargs['feature_net_path'] + 'layers.pickle', 'rb'))
                    contact_pos = contact_info[:, :, -9:-6]
                    self.features = supervised_nn_max_pool(contact_pos, pos_layers[0], pos_layers[1] + [3], name = 'predicted_pos', reuse=tf.AUTO_REUSE)
                    real_pos = object_info[:, -7:-4]

                    predicted_o = tf.concat([object_info[:, :-7], self.features, object_info[:, -4:]], axis = -1)
                    o = self.o_stats.normalize(predicted_o)
            elif self.pre_train == 'cpc':
                with tf.variable_scope('pi/process'):
                    z_dim, fixed_num_of_contact, contact_point_dim, action_dim, encoder_lr, feature_dims, trans_mode, label, include_action = pickle.load(open(self.feature_net_path + 'params.pickle', 'rb'))
                    features = to_latent(x = contact_info, z_dim = z_dim, layers_sizes = [32, 32], name='new_cpc', process_type = self.process_type)
                    self.features = tf.clip_by_value(features, -self.clip_obs, self.clip_obs)
                    features = self.feature_stats.normalize(features)
                    object_info = self.o_stats.normalize(object_info)
                    o = tf.concat([features, object_info], axis = -1)
            else:
                o = self.o_stats.normalize(self.o_tf)
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

    def _get_q_target(self):
        next_observations_ph = self.op_phs_dict['next_observations']
        dist_info_sym = self.policy.distribution_info_sym(next_observations_ph)
        next_actions_var, dist_info_sym = self.policy.distribution.sample_sym(dist_info_sym)
        next_log_pis_var = self.policy.distribution.log_likelihood_sym(next_actions_var, dist_info_sym)
        next_log_pis_var = tf.expand_dims(next_log_pis_var, axis=-1)

        input_q_fun = tf.concat([next_observations_ph, next_actions_var], axis=-1)
        next_q_values = [Q.value_sym(input_var=input_q_fun) for Q in self.Q_targets]

        min_next_Q = tf.reduce_min(next_q_values, axis=0)
        next_values_var = min_next_Q - self.alpha * next_log_pis_var

        dones_ph = tf.cast(self.op_phs_dict['dones'], next_values_var.dtype)
        dones_ph = tf.expand_dims(dones_ph, axis=-1)
        rewards_ph = self.op_phs_dict['rewards']
        rewards_ph = tf.expand_dims(rewards_ph, axis=-1)
        self.q_target = td_target(
            reward=self.reward_scale * rewards_ph,
            discount=self.discount,
            next_value=(1 - dones_ph) * next_values_var)

        return tf.stop_gradient(self.q_target)

    def _init_critic_update(self):
        """Create minimization operation for critic Q-function.
        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.
        See Equations (5, 6) in [1], for further information of the
        Q-function update rule.
        """
        q_target = self._get_q_target()
        assert q_target.shape.as_list() == [None, 1]
        observations_ph = self.op_phs_dict['observations']
        actions_ph = self.op_phs_dict['actions']
        input_q_fun = tf.concat([observations_ph, actions_ph], axis=-1)

        q_values_var = self.q_values_var = [Q.value_sym(input_var=input_q_fun) for Q in self.Qs]
        q_losses = self.q_losses = [tf.compat.v1.losses.mean_squared_error(labels=q_target, predictions=q_value, weights=0.5)
                                    for q_value in q_values_var]

        self.q_optimizers = [tf.train.AdamOptimizer(
                                                    learning_rate=self.Q_lr,
                                                    name='{}_{}_optimizer'.format(Q.name, i)
                                                    )
                             for i, Q in enumerate(self.Qs)]

        q_training_ops = [
            q_optimizer.minimize(loss=q_loss, var_list=list(Q.vfun_params.values()))
            for i, (Q, q_loss, q_optimizer)
            in enumerate(zip(self.Qs, q_losses, self.q_optimizers))]

        self.training_ops.update({'Q': tf.group(q_training_ops)})

    def _init_actor_update(self):
        """Create minimization operations for policy and entropy.
        Creates a `tf.optimizer.minimize` operations for updating
        policy and entropy with gradient descent, and adds them to
        `self._training_ops` attribute.
        See Section 4.2 in [1], for further information of the policy update,
        and Section 5 in [1] for further information of the entropy update.
        """
        observations_ph = self.op_phs_dict['observations']
        dist_info_sym = self.policy.distribution_info_sym(observations_ph)
        actions_var, dist_info_sym = self.policy.distribution.sample_sym(dist_info_sym)
        log_pis_var = self.policy.distribution.log_likelihood_sym(actions_var, dist_info_sym)
        log_pis_var = tf.expand_dims(log_pis_var, axis=1)

        assert log_pis_var.shape.as_list() == [None, 1]

        log_alpha = tf.get_variable('log_alpha', dtype=tf.float32, initializer=0.0)
        alpha = tf.exp(log_alpha)

        if isinstance(self.target_entropy, Number):
            alpha_loss = -tf.reduce_mean(
                log_alpha * tf.stop_gradient(log_pis_var + self.target_entropy))
            self.log_pis_var = log_pis_var

            self.alpha_optimizer = tf.train.AdamOptimizer(self.policy_lr, name='alpha_optimizer')
            self.alpha_train_op = self.alpha_optimizer.minimize(loss=alpha_loss, var_list=[log_alpha])

            self.training_ops.update({
                'temperature_alpha': self.alpha_train_op
            })

        self.alpha = alpha

        if self.action_prior == 'normal':
            raise NotImplementedError
        elif self.action_prior == 'uniform':
            policy_prior_log_probs = 0.0

        input_q_fun = tf.concat([observations_ph, actions_var], axis=-1)
        next_q_values = [Q.value_sym(input_var=input_q_fun) for Q in self.Qs]
        min_q_val_var = tf.reduce_min(next_q_values, axis=0)

        if self.reparameterize:
            policy_kl_losses = (self.alpha * log_pis_var - min_q_val_var - policy_prior_log_probs)
        else:
            raise NotImplementedError

        assert policy_kl_losses.shape.as_list() == [None, 1]

        self.policy_losses = policy_kl_losses
        policy_loss = tf.reduce_mean(policy_kl_losses)

        self.policy_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.policy_lr,
            name="policy_optimizer")

        policy_train_op = self.policy_optimizer.minimize(
            loss=policy_loss,
            var_list=list(self.policy.policy_params.values()))

        self.training_ops.update({'policy_train_op': policy_train_op})
