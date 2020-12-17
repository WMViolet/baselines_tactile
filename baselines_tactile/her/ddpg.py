from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea

from tactile_baselines import logger
from tactile_baselines.her.supervised.model import FeatureNet
from tactile_baselines.her.util import (
    import_function, store_args, flatten_grads, transitions_in_episode_batch, convert_episode_to_batch_major)
from tactile_baselines.her.normalizer import Normalizer
from tactile_baselines.her.tf_normalizer import TfNormalizer
from tactile_baselines.her.replay_buffer import ReplayBuffer
from tactile_baselines.common.mpi_adam import MpiAdam
from tactile_baselines.common import tf_util
from pdb import set_trace as st
from tactile_baselines.cpc.cpc_model_tf import *
import dill as pickle
import time


def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}


# global DEMO_BUFFER #buffer for demonstrations
global test_sess_time

class DDPG(object):
    @store_args
    def __init__(self, input_dims, buffer_size, hidden, layers, network_class, polyak, batch_size,
                 Q_lr, pi_lr, norm_eps, norm_clip, max_u, action_l2, clip_obs, scope, T,
                 rollout_batch_size, subtract_goals, relative_goals, clip_pos_returns, clip_return,
                 bc_loss, q_filter, num_demo, demo_batch_size, prm_loss_weight, aux_loss_weight,
                 sample_transitions, gamma, reuse=False, pre_train_model=False, update_model=True, feature_net_path='', **kwargs):
        """Implementation of DDPG that is used in combination with Hindsight Experience Replay (HER).
            Added functionality to use demonstrations for training to Overcome exploration problem.
        Args:
            input_dims (dict of ints): dimensions for the observation (o), the goal (g), and the
                actions (u)
            buffer_size (int): number of transitions that are stored in the replay buffer
            hidden (int): number of units in the hidden layers
            layers (int): number of hidden layers
            network_class (str): the network class that should be used (e.g. 'baselines.her.ActorCritic')
            polyak (float): coefficient for Polyak-averaging of the target network
            batch_size (int): batch size for training
            Q_lr (float): learning rate for the Q (critic) network
            pi_lr (float): learning rate for the pi (actor) network
            norm_eps (float): a small value used in the normalizer to avoid numerical instabilities
            norm_clip (float): normalized inputs are clipped to be in [-norm_clip, norm_clip]
            max_u (float): maximum action magnitude, i.e. actions are in [-max_u, max_u]
            action_l2 (float): coefficient for L2 penalty on the actions
            clip_obs (float): clip observations before normalization to be in [-clip_obs, clip_obs]
            scope (str): the scope used for the TensorFlow graph
            T (int): the time horizon for rollouts
            rollout_batch_size (int): number of parallel rollouts per DDPG agent
            subtract_goals (function): function that subtracts goals from each other
            relative_goals (boolean): whether or not relative goals should be fed into the network
            clip_pos_returns (boolean): whether or not positive returns should be clipped
            clip_return (float): clip returns to be in [-clip_return, clip_return]
            sample_transitions (function) function that samples from the replay buffer
            gamma (float): gamma used for Q learning updates
            reuse (boolean): whether or not the networks should be reused
            bc_loss: whether or not the behavior cloning loss should be used as an auxiliary loss
            q_filter: whether or not a filter on the q value update should be used when training with demonstartions
            num_demo: Number of episodes in to be used in the demonstration buffer
            demo_batch_size: number of samples to be used from the demonstrations buffer, per mpi thread
            prm_loss_weight: Weight corresponding to the primary loss
            aux_loss_weight: Weight corresponding to the auxiliary loss also called the cloning loss
        """
        if self.clip_return is None:
            self.clip_return = np.inf

        # ADDED
        self.use_contact = (self.contact_dim > 0)
        self.pre_train_model = pre_train_model
        self.feature_net_path = feature_net_path
        self.process_type = kwargs['process_type']
        self.contact_dim = kwargs['contact_dim']
        self.__dict__['use_contact'] = self.use_contact
        self.__dict__['pre_train'] = self.pre_train_model

        self.create_actor_critic = import_function(self.network_class)
        input_shapes = dims_to_shapes(self.input_dims)
        self.dimo = self.input_dims['o'] - self.contact_dim
        self.dimg = self.input_dims['g']
        self.dimu = self.input_dims['u']
        self.feature_dim = kwargs['feature_dim']
        self.contact_point_dim = self.contact_dim // self.fixed_num_of_contact


        # Prepare staging area for feeding data to the model.
        stage_shapes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                continue
            stage_shapes[key] = (None, *input_shapes[key])
        for key in ['o', 'g']:
            stage_shapes[key + '_2'] = stage_shapes[key]
        stage_shapes['r'] = (None,)
        self.stage_shapes = stage_shapes

        # Create network.
        with tf.variable_scope(self.scope):
            logger.info("Creating a DDPG agent with action space %d x %s..." % (self.dimu, self.max_u))
            self.sess = tf_util.get_session()
            # order: ['g', 'o', 'u', 'o_2', 'g_2', 'r'])
            if self.pre_train_model == 'cpc':
                self.staging_tf = StagingArea(
                    dtypes=[tf.float32 for _ in self.stage_shapes.keys()],
                    shapes=list(self.stage_shapes.values()))
                self.buffer_ph_tf = [tf.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes.values()]
                self.stage_op = self.staging_tf.put(self.buffer_ph_tf)

                self.cpc_shape = OrderedDict()
                self.cpc_shape['obs_neg'] = (None, self.fixed_num_of_contact, self.contact_point_dim)
                self.cpc_shape['obs_pos'] = (None, self.fixed_num_of_contact, self.contact_point_dim)
                self.cpc_staging_tf = StagingArea(
                    dtypes=[tf.float32 for _ in self.cpc_shape.keys()],
                    shapes=list(self.cpc_shape.values()))
                self.cpc_buffer_ph_tf = [tf.placeholder(tf.float32, shape=shape) for shape in self.cpc_shape.values()]
                self.cpc_stage_op = self.cpc_staging_tf.put(self.cpc_buffer_ph_tf)
            else:
                self.staging_tf = StagingArea(
                    dtypes=[tf.float32 for _ in self.stage_shapes.keys()],
                    shapes=list(self.stage_shapes.values()))
                self.buffer_ph_tf = [tf.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes.values()]
                self.stage_op = self.staging_tf.put(self.buffer_ph_tf)
            self.update_model = update_model

            if self.pre_train_model != 'none':
                self.__dict__['feature_net_path'] = self.feature_net_path
                self.__dict__['clip_obs'] = self.clip_obs


            self._create_network(reuse=reuse)

        # Configure the replay buffer.
        buffer_shapes = {key: (self.T-1 if key != 'o' else self.T, *input_shapes[key])
                         for key, val in input_shapes.items()}
        buffer_shapes['g'] = (buffer_shapes['g'][0], self.dimg)
        buffer_shapes['ag'] = (self.T, self.dimg)

        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size
        self.buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions)


    def _random_action(self, n):
        return np.random.uniform(low=-self.max_u, high=self.max_u, size=(n, self.dimu))


    def _preprocess_og(self, o, ag, g):
        # self.clip_obs = 200
        if self.relative_goals:
            g_shape = g.shape
            g = g.reshape(-1, self.dimg)
            ag = ag.reshape(-1, self.dimg)
            g = self.subtract_goals(g, ag)
            g = g.reshape(*g_shape)
        if len(o.shape) == 1:
            o[-self.dimo:] = np.clip(o[-self.dimo:], -self.clip_obs, self.clip_obs)
        elif len(o.shape) == 2:
            o[:, -self.dimo:] = np.clip(o[:, -self.dimo:], -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, g

    def step(self, obs):
        actions = self.get_actions(obs['observation'], obs['achieved_goal'], obs['desired_goal'])
        return actions, None, None, None


    def get_actions(self, o, ag, g, noise_eps=0., random_eps=0., use_target_net=False,
                    compute_Q=False):
        o, g = self._preprocess_og(o, ag, g)
        policy = self.target if use_target_net else self.main
        # values to compute
        vals = [policy.pi_tf]
        if compute_Q:
            vals += [policy.Q_pi_tf]

        """lines added here, remove later"""
        ori = o[:,-7:-4].copy()
        noise = np.random.normal(0, 7e-4, ori.shape)
        o[:,-7:-4] += noise

        feed = {
            policy.o_tf: o.reshape(-1, self.dimo+self.contact_dim),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: np.zeros((o.size // (self.dimo+self.contact_dim), self.dimu), dtype=np.float32)
        }

        ret = self.sess.run(vals, feed_dict=feed)

        u = ret[0]
        noise = noise_eps * self.max_u * np.random.randn(*u.shape)  # gaussian noise
        u += noise
        u = np.clip(u, -self.max_u, self.max_u)
        u += np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (self._random_action(u.shape[0]) - u)  # eps-greedy
        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()
        ret[0] = u

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def store_episode(self, episode_batch, update_stats=True):
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key
                       'o' is of size T+1, others are of size T
        """

        """lines added, remove later"""
        ori = episode_batch['o'][:,:,-7:-4].copy()
        noise = np.random.normal(0, 7e-4, ori.shape)
        episode_batch['o'][:,:,-7:-4] += noise

        self.buffer.store_episode(episode_batch)
        if update_stats:
            # add transitions to normalizer
            episode_batch['o_2'] = episode_batch['o'][:, 1:, :]
            episode_batch['ag_2'] = episode_batch['ag'][:, 1:, :]
            num_normalizing_transitions = transitions_in_episode_batch(episode_batch)
            # change goals here, recompute rewards
            transitions = self.sample_transitions(episode_batch, num_normalizing_transitions)

            o, g, ag = transitions['o'], transitions['g'], transitions['ag']
            transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
            # No need to preprocess the o_2 and g_2 since this is only used for stat
            """Normalization stuff here. """
            self.o_stats.update(transitions['o'][:, -self.o_stats.size:])
            self.g_stats.update(transitions['g'])
            if self.pre_train_model in ['cpc', 'curl']:
                feed_dict = {self.main.o_tf:transitions['o']}
                features = self.sess.run(self.main.features, feed_dict=feed_dict)
                features = np.clip(features, -self.clip_obs, self.clip_obs)
                self.feature_stats.update(features)
                self.feature_stats.recompute_stats()
            # elif self.process_type == 'max_pool':
            #     feed_dict = {self.main.o_tf:transitions['o']}
            #     features = self.sess.run(self.main.features, feed_dict=feed_dict)
            #     self.feature_stats.update(features)
            #     self.feature_stats.recompute_stats()



            self.o_stats.recompute_stats()
            self.g_stats.recompute_stats()
        return transitions['o']


    def get_current_buffer_size(self):
        return self.buffer.get_current_size()

    def _sync_optimizers(self):
        self.Q_adam.sync()
        self.pi_adam.sync()
        if self.pre_train_model == 'supervised':
            self.feature_adam.sync()
        elif self.pre_train_model == 'cpc':
            self.cpc.sync()
        elif self.pre_train_model == 'curl':
            self.curl_adam.sync()
            self.encoder_adam.sync()

    def _grads(self):
        # Avoid feed_dict here for performance!
        critic_loss, actor_loss, Q_grad, pi_grad = self.sess.run([
            self.Q_loss_tf,
            self.main.Q_pi_tf,
            self.Q_grad_tf,
            self.pi_grad_tf
        ])
        return critic_loss, actor_loss, Q_grad, pi_grad

    def _update(self, Q_grad, pi_grad):
        self.Q_adam.update(Q_grad, self.Q_lr)
        self.pi_adam.update(pi_grad, self.pi_lr)

    def sample_batch(self):
        transitions = self.buffer.sample(self.batch_size) #otherwise only sample from primary buffer
        o, o_2, g = transitions['o'], transitions['o_2'], transitions['g']
        ag, ag_2 = transitions['ag'], transitions['ag_2']
        transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
        transitions['o_2'], transitions['g_2'] = self._preprocess_og(o_2, ag_2, g)

        transitions_batch = [transitions[key] for key in self.stage_shapes.keys() if key not in ['obs_pos', 'obs_neg']]
        return transitions_batch


    def stage_batch(self, batch=None):
        if batch is None:
            batch = self.sample_batch()
        assert len(self.buffer_ph_tf) == len(batch)

        """lines added, remove them later"""
        ori = batch[1][:,-7:-4].copy()
        noise = np.random.normal(0, 7e-4, ori.shape)
        batch[1][:,-7:-4] += noise
        noise = np.random.normal(0, 7e-4, ori.shape)
        batch[3][:,-7:-4] += noise

        self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))

        if self.pre_train_model == 'supervised':
            assert batch[1].shape[1] == 583, "must use full observations"
            # 253, 251, 246, 233, 232, 220, 215, 210
            # feature_loss, max_feature_loss, feature_grad = self.sess.run([self.feature_loss_tf, self.max_feature_loss, self.feature_grad_tf])
            feature_loss, feature_grad = self.sess.run([self.feature_loss_tf, self.feature_grad_tf])
            self.feature_adam.update(feature_grad, 1e-3)
            self.sess.run(self.update_feature_weights_target)
            self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))
            # writer = tf.summary.FileWriter("home/vioichigo/try/tactile-baselines/graph", self.sess.graph)
            # print(self.sess.run(self.main.features))
            # writer.close()
            return feature_loss
        elif self.pre_train_model == 'cpc':
            # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # run_metadata = tf.RunMetadata()
            # obs = pickle.load(open('/home/vioichigo/try/tactile-baselines/dataset/HandManipulateEgg-v0/50000obs.pickle', 'rb'))
            # indices = np.random.randint(obs.shape[0], size=batch[1].shape[0] * (self.main.n_neg - 1))
            # obs_neg = obs[indices]
            # obs_pos = batch[3][:, :self.contact_dim].reshape((-1, self.fixed_num_of_contact, self.contact_dim//self.fixed_num_of_contact))
            # # self.sess.run(self.cpc_stage_op, feed_dict=dict(zip(self.cpc_buffer_ph_tf, [obs_neg, obs_pos])), options=run_options, run_metadata=run_metadata)
            # first = time.time()
            # # self.sess.run(self.cpc_stage_op, feed_dict=dict(zip(self.cpc_buffer_ph_tf, [obs_neg, obs_pos])))
            # start = time.time()
            # print("feed:", start - first)
            # feed_dict = {self.cpc_inputs_tf['obs_pos']: obs_pos, self.cpc_inputs_tf['obs_neg']: obs_neg}
            # # dict(zip(self.cpc_inputs_tf, [obs_neg, obs_pos]))
            # cpc_loss, cpc_grad = self.sess.run([self.cpc_loss_tf, self.cpc_grad_tf], feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
            # tl = timeline.Timeline(run_metadata.step_stats)
            # ctf = tl.generate_chrome_trace_format()
            # with open('./timeline.json', 'w') as f:
            #     f.write(ctf)
            # now = time.time()
            # print("compute_loss", now - start)
            # self.cpc_adam.update(cpc_grad, 1e-3)
            # print("update weights", time.time() - now)
            # self.sess.run(self.update_cpc_weights_target)
            # self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))

            return 1
        elif self.pre_train_model == 'curl':
            curl_loss, curl_grad, encoder_grad = self.sess.run([self.curl_loss, self.curl_grad_tf, self.encoder_grad_tf])
            self.curl_adam.update(curl_grad, 1e-3)
            self.encoder_adam.update(encoder_grad, 1e-3)

            self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))
            self.sess.run(self.update_curl_weights_op)

            return curl_loss


            # return cpc_loss





    def train(self, stage=True):
        if stage:
            if self.pre_train_model == 'none':
                self.stage_batch()
            else:
                feature_loss = self.stage_batch()
        critic_loss, actor_loss, Q_grad, pi_grad = self._grads()
        self._update(Q_grad, pi_grad)
        if self.pre_train_model == 'none':
            return critic_loss, actor_loss
        else:
            return critic_loss, actor_loss, feature_loss

    def _init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def update_target_net(self):
        self.sess.run(self.update_target_net_op)

    def clear_buffer(self):
        self.buffer.clear_buffer()

    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        if self.pre_train_model == 'supervised':
            if not self.update_model:
                res = [x for x in res if x.name.find('predicted_pos') == -1]
        elif self.pre_train_model == 'cpc':
            if not self.update_model:
                res = [x for x in res if x.name.find('new_cpc') == -1]
        # elif self.pre_train_model == 'curl':
        #     res = [x for x in res if x.name.find('W') == -1]
        assert len(res) > 0
        return res

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        return res

    def _create_network(self, reuse=False):
        # running averages
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.increment_global_step = tf.assign_add(self.global_step, 1, name = 'increment_global_step')
        with tf.variable_scope('o_stats') as vs:
            if reuse:
                vs.reuse_variables()
            """Normalization stuff here. """
            if self.use_contact and self.process_type in ['none', 'test']:
                self.o_stats = Normalizer(self.dimo+self.contact_dim, self.norm_eps, self.norm_clip, sess=self.sess)
            else:
                self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope('g_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)


        if self.pre_train_model == 'cpc':
            with tf.variable_scope('feature_stats') as vs:
                if reuse:
                    vs.reuse_variables()
                z_dim = pickle.load(open(self.feature_net_path + 'params.pickle', 'rb'))[0]
                self.feature_stats = Normalizer(z_dim, self.norm_eps, self.norm_clip, sess=self.sess)
                self.__dict__['feature_normalizer'] = self.feature_stats
        elif self.pre_train_model == 'curl':
            with tf.variable_scope('feature_stats') as vs:
                if reuse:
                    vs.reuse_variables()
                self.feature_stats = Normalizer(32, self.norm_eps, self.norm_clip, sess=self.sess)
                self.__dict__['feature_normalizer'] = self.feature_stats


        # mini-batch sampling.
        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i])
                                for i, key in enumerate(self.stage_shapes.keys())])
        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1])

        if self.pre_train_model == 'cpc':
            cpc_batch = self.cpc_staging_tf.get()
            cpc_batch_tf = OrderedDict([(key, cpc_batch[i])
                                    for i, key in enumerate(self.cpc_shape.keys())])
            # self.cpc_batch_tf = {}
            # self.cpc_batch_tf['obs_neg'] = tf.placeholder(tf.float32, shape=(None, self.fixed_num_of_contact, self.contact_point_dim))
            # self.cpc_batch_tf['obs_pos'] = tf.placeholder(tf.float32, shape=(None, self.fixed_num_of_contact, self.contact_point_dim))
            # self.__dict__['cpc_inputs_tf'] = self.cpc_batch_tf

        #choose only the demo buffer samples

        # networks
        with tf.variable_scope('main') as vs:
            if reuse:
                vs.reuse_variables()
            self.main = self.create_actor_critic(batch_tf, net_type='main', **self.__dict__)
            vs.reuse_variables()
        with tf.variable_scope('target') as vs:
            # reuse = False
            if reuse:
                vs.reuse_variables()
            target_batch_tf = batch_tf.copy()
            target_batch_tf['o'] = batch_tf['o_2'] #next_observations
            target_batch_tf['g'] = batch_tf['g_2'] #next_goals
            self.target = self.create_actor_critic(
                target_batch_tf, net_type='target', **self.__dict__)
            vs.reuse_variables()

        assert len(self._vars("main")) == len(self._vars("target"))

        # loss functions
        target_Q_pi_tf = self.target.Q_pi_tf
        clip_range = (-self.clip_return, 0. if self.clip_pos_returns else np.inf)
        target_tf = tf.clip_by_value(batch_tf['r'] + self.gamma * target_Q_pi_tf, *clip_range)
        self.Q_loss_tf = tf.reduce_mean(tf.square(tf.stop_gradient(target_tf) - self.main.Q_tf))

        # else: #If  not training with demonstrations
        self.pi_loss_tf = -tf.reduce_mean(self.main.Q_pi_tf)
        self.pi_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))

        if self.pre_train_model == 'supervised':
            self.feature_net_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ddpg/main/pi/process/predicted_pos')
            pos = batch_tf['o'][:, self.contact_dim:][:, -7:-4]
            self.feature_loss_tf = tf.reduce_mean(tf.square(pos - self.main.features))
            # self.max_feature_loss = tf.reduce_max(tf.square(pos - self.main.features))
            feature_grads_tf = tf.gradients(self.feature_loss_tf, self.feature_net_var)
            assert len(self.feature_net_var) == len(feature_grads_tf)
            self.feature_grads_vars_tf = zip(feature_grads_tf, self.feature_net_var)
            self.feature_grad_tf = flatten_grads(grads=feature_grads_tf, var_list=self.feature_net_var)
            self.feature_adam = MpiAdam(self.feature_net_var, scale_grad_by_procs=False)

            target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ddpg/target/pi/process/predicted_pos')
            self.update_feature_weights_target = [tf.assign(new, old) for (new, old) in zip(target_vars, self.feature_net_var)]
        elif self.pre_train_model == 'cpc':
            self.cpc_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ddpg/main/pi/process/new_cpc')
            pos = tf.reshape(batch_tf['o_2'][:, :self.contact_dim], [-1, self.fixed_num_of_contact, self.contact_dim//self.fixed_num_of_contact])
            with tf.variable_scope('auxiliary'):
                self.cpc_loss_tf = compute_cpc_loss(self.main.z_dim, self.main.pos_features, self.main.neg_features, self.main.next,
                                                  process_type = self.process_type, n_neg = self.main.n_neg, type = self.main.type)
            cpc_grads_tf = tf.gradients(self.cpc_loss_tf, self.cpc_var)
            assert len(self.cpc_var) == len(cpc_grads_tf)
            self.cpc_grads_vars_tf = zip(cpc_grads_tf, self.cpc_var)
            self.cpc_grad_tf = flatten_grads(grads=cpc_grads_tf, var_list=self.cpc_var)
            self.cpc_adam = MpiAdam(self.cpc_var, scale_grad_by_procs=False)

            target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ddpg/target/pi/process/new_cpc')
            self.update_cpc_weights_target = [tf.assign(new, old) for (new, old) in zip(target_vars, self.cpc_var)]

        elif self.pre_train_model == 'curl':
            self.W = tf.get_variable("W", shape=[self.main.z_dim, self.main.z_dim], trainable=True)
            self.encoder_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ddpg/main/pi/curl')
            self.encoder_adam = MpiAdam(self.encoder_var, scale_grad_by_procs=False)
            self.curl_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ddpg/main/pi/curl') + [self.W]
             # + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ddpg/target/pi/curl')
            self.curl_adam = MpiAdam(self.curl_var, scale_grad_by_procs=False)


            z_a = self.main.features
            z_pos = tf.stop_gradient(self.target.features)

            Wz = tf.matmul(self.W, tf.transpose(z_pos))  # (z_dim,B)
            logits = tf.matmul(z_a, Wz)  # (B,B)
            logits = logits - tf.reduce_max(logits, 1)[:, None]
            labels = tf.range(tf.shape(logits)[0])
            self.curl_loss = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels)

            target_curl_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ddpg/target/pi/curl')

            self.update_curl_weights_op = list(
                map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(target_curl_vars, self.encoder_var)))

            curl_grads_tf = tf.gradients(self.curl_loss, self.curl_var)
            self.curl_grad_tf = flatten_grads(grads=curl_grads_tf, var_list=self.curl_var)
            encoder_grads_tf = tf.gradients(self.curl_loss, self.encoder_var)
            self.encoder_grad_tf = flatten_grads(grads=encoder_grads_tf, var_list=self.encoder_var)


        Q_grads_tf = tf.gradients(self.Q_loss_tf, self._vars('main/Q'))
        pi_grads_tf = tf.gradients(self.pi_loss_tf, self._vars('main/pi'))
        assert len(self._vars('main/Q')) == len(Q_grads_tf)
        assert len(self._vars('main/pi')) == len(pi_grads_tf)
        self.Q_grads_vars_tf = zip(Q_grads_tf, self._vars('main/Q'))
        self.pi_grads_vars_tf = zip(pi_grads_tf, self._vars('main/pi'))
        self.Q_grad_tf = flatten_grads(grads=Q_grads_tf, var_list=self._vars('main/Q'))
        self.pi_grad_tf = flatten_grads(grads=pi_grads_tf, var_list=self._vars('main/pi'))

        # optimizers
        self.Q_adam = MpiAdam(self._vars('main/Q'), scale_grad_by_procs=False)
        self.pi_adam = MpiAdam(self._vars('main/pi'), scale_grad_by_procs=False)

        # polyak averaging
        self.main_vars = self._vars('main/Q') + self._vars('main/pi')
        self.target_vars = self._vars('target/Q') + self._vars('target/pi')

        self.stats_vars = self._global_vars('o_stats') + self._global_vars('g_stats')

        self.init_target_net_op = list(
            map(lambda v: v[0].assign(v[1]), zip(self.target_vars, self.main_vars)))
        self.update_target_net_op = list(
            map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(self.target_vars, self.main_vars)))

        # initialize all variables
        tf.variables_initializer(self._global_vars('')).run()
        self._sync_optimizers()
        self._init_target_net()



    def logs(self, prefix=''):
        logs = []
        logs += [('stats_o/mean', np.mean(self.sess.run([self.o_stats.mean])))]
        logs += [('stats_o/std', np.mean(self.sess.run([self.o_stats.std])))]
        logs += [('stats_g/mean', np.mean(self.sess.run([self.g_stats.mean])))]
        logs += [('stats_g/std', np.mean(self.sess.run([self.g_stats.std])))]

        if prefix != '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def __getstate__(self):
        """Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        excluded_subnames = ['_tf', '_op', '_vars', '_adam', 'buffer', 'sess', '_stats',
                             'main', 'target', 'lock', 'env', 'sample_transitions',
                             'stage_shapes', 'create_actor_critic']

        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        state['buffer_size'] = self.buffer_size
        state['tf'] = self.sess.run([x for x in self._global_vars('') if 'buffer' not in x.name])
        return state

    def __setstate__(self, state):
        if 'sample_transitions' not in state:
            # We don't need this for playing the policy.
            state['sample_transitions'] = None

        self.__init__(**state)
        # set up stats (they are overwritten in __init__)
        for k, v in state.items():
            if k[-6:] == '_stats':
                self.__dict__[k] = v
        # load TF variables
        vars = [x for x in self._global_vars('') if 'buffer' not in x.name]
        assert(len(vars) == len(state["tf"]))
        node = [tf.assign(var, val) for var, val in zip(vars, state["tf"])]
        self.sess.run(node)

    def save(self, save_path):
        tf_util.save_variables(save_path)
