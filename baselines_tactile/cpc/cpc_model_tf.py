import tensorflow as tf
from pdb import set_trace as st
from tactile_baselines import logger
import numpy as np
from tensorflow.python.client import timeline
import time 

def to_latent(x, z_dim, layers_sizes, name, reuse = tf.AUTO_REUSE, process_type = 'max_pool'):
    if process_type == 'none':
        assert len(x.shape) == 3
        x = tf.reshape(x, (-1, np.prod(x.shape[1:])))
    for i, size in enumerate(layers_sizes):
        activation = tf.nn.relu
        x = tf.layers.dense(inputs=x,
                            units=size,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            reuse=reuse,
                            name=name + '_' + str(i))
        x = activation(x)
    z = tf.layers.dense(inputs=x,
                        units=z_dim,
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        reuse=reuse,
                        name=name + '_' + str(len(layers_sizes)))
    if process_type == 'max_pool':
        z = tf.reduce_max(z, axis = 1)
    return z




def compute_cpc_loss(z_dim, z_pos, z_neg, z_next, process_type = 'max_pool', n_neg = 100, type = 2):
    """use the same model"""
    if type == 1:
        z_next = tf.reshape(z_next, [-1, 1, z_dim])  # b x 1 x z_dim
        z_pos = tf.reshape(z_pos, [-1, z_dim, 1]) # b x z_dim x 1
        pos_log_density = tf.squeeze(tf.matmul(z_next, z_pos), axis = -1) # b x 1
        batch_size = tf.shape(z_next)[0]

        z_neg = tf.tile(tf.expand_dims(tf.transpose(z_neg), axis = 0), [batch_size, 1, 1]) # b x z_dim x n
        neg_log_density = tf.squeeze(tf.matmul(z_next, z_neg), axis = 1) # b x n

        loss = tf.concat((tf.zeros((batch_size, 1)), neg_log_density - pos_log_density), axis=1)  # b x n+1
        loss = tf.reduce_mean(tf.math.reduce_logsumexp(loss, axis = 1))
    elif type == 2:
        pos_log_density = tf.reduce_sum(z_next * z_pos, axis = 1) # b x z_dim
        pos_log_density = -tf.reduce_sum((z_next ** 2), axis = 1) - 2 * pos_log_density + tf.reduce_sum((z_pos ** 2), axis=1)
        pos_log_density = tf.expand_dims(pos_log_density, axis = 1) # b x 1

        z_neg = tf.reshape(z_neg, [-1, (n_neg-1), z_dim])
        bs = tf.shape(z_neg)[0]
        z_next = tf.expand_dims(z_next, axis = 1)
        neg_log_density = tf.squeeze(tf.matmul(z_next, tf.transpose(z_neg, [0, 2, 1])), axis = 1)  # b x n
        neg_log_density = -tf.reduce_sum((z_next ** 2), axis = -1) - 2 * neg_log_density + tf.reduce_sum((z_neg ** 2), axis =-1)

        loss = tf.concat((tf.zeros((bs, 1)), neg_log_density - pos_log_density), axis = 1)
        loss = tf.reduce_mean(tf.reduce_logsumexp(loss, axis = 1))
    return loss


def predict(x, name, z_dim, mode = 'MLP', reuse = None):
    if mode == 'MLP':
        hidden_sizes = [32, 32]
        for i, size in enumerate(hidden_sizes):
            activation = tf.nn.relu
            x = tf.layers.dense(inputs=x,
                                units=size,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                reuse=reuse,
                                name=name + '_' + str(i))
            x = activation(x)
        x = tf.layers.dense(inputs=x,
                            units=z_dim,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            reuse=reuse,
                            name=name + '_' + str(len(hidden_sizes)))
    elif mode == 'linear':
        x = tf.layers.dense(inputs=x, #[B, z_dim + action_dim]
                            units=z_dim,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            reuse=reuse,
                            use_bias=False,
                            name=name)
    return x



class Encoder:
    def __init__(self, z_dim,
                       fixed_num_of_contact,
                       contact_point_dim,
                       feature_dims):
        self.z_dim = z_dim
        self.fixed_num_of_contact = fixed_num_of_contact
        self.feature_dims = feature_dims

    def to_latent(self, x, name, reuse = tf.AUTO_REUSE, process_type = 'max_pool'):
        return to_latent(x, self.z_dim, self.feature_dims, name, reuse=reuse, process_type = process_type)
        # <tf.Variable 'cpc_0/kernel:0' shape=(58, 32) dtype=float32_ref>
        # <tf.Variable 'cpc_0/bias:0' shape=(32,) dtype=float32_ref>
        # <tf.Variable 'cpc_1/kernel:0' shape=(32, 32) dtype=float32_ref>
        # <tf.Variable 'cpc_1/bias:0' shape=(32,) dtype=float32_ref>
        # <tf.Variable 'cpc_2/kernel:0' shape=(32, 32) dtype=float32_ref>
        # <tf.Variable 'cpc_2/bias:0' shape=(32,) dtype=float32_ref>
        # return z


class Transition:
    def __init__(self, z_dim, action_dim, mode = 'linear'):
        self.z_dim = z_dim
        self.mode = mode

    def predict(self, x, name, reuse = None):
        return predict(x, name, self.z_dim, mode = 'MLP', reuse = reuse)


class CPC:
    def __init__(self, sess,
                       encoder,
                       trans,
                       encoder_lr,
                       fixed_num_of_contact,
                       contact_point_dim,
                       action_dim,
                       include_action=True,
                       type=1,
                       n_neg=50,
                       process_type='max_pool',
                       mode = 'train'
                       ):
        self.encoder = encoder
        self.trans = trans
        self.sess = sess
        self.z_dim = self.encoder.z_dim
        self.include_action = include_action
        self.type = type
        self.n_neg = n_neg
        self.process_type = process_type

        if mode == 'train':
            self.obs = tf.placeholder(tf.float32, [None, fixed_num_of_contact, contact_point_dim], name = 'obs')
            self.obs_pos = tf.placeholder(tf.float32, [None, fixed_num_of_contact, contact_point_dim], name = 'obs_pos')
            self.obs_neg = tf.placeholder(tf.float32, [None, fixed_num_of_contact, contact_point_dim], name = 'obs_neg')
            self.actions = tf.placeholder(tf.float32, [None, action_dim], name = 'actions')
            self.z = self.encoder.to_latent(self.obs, name = 'cpc', process_type = self.process_type) # b x z_dim
            z_pos = self.encoder.to_latent(self.obs_pos, name = 'cpc', process_type = self.process_type)  # b x z_dim
            z_neg = self.encoder.to_latent(self.obs_neg, name = 'cpc', process_type = self.process_type)  # n x z_dim
            self.z = tf.identity(self.z, name = 'z')
            if self.include_action:
                assert self.actions != None, 'must feed actions'
                z = tf.concat((self.z, self.actions), axis=1)
            else:
                z = self.z
            z_next = trans.predict(z, name = 'trans_graph')  # b x z_dim
            self.loss = compute_cpc_loss(self.encoder.z_dim, z_pos, z_neg, z_next,
                                         process_type = process_type, n_neg = self.n_neg, type = self.type)
            self.loss = tf.identity(self.loss, name = 'loss')
            self.optimizer = tf.train.AdamOptimizer(learning_rate=encoder_lr)
            self.op = self.optimizer.minimize(self.loss)
        elif mode in  ['restore', 'store_weights']:
            self.restore_encoder()


    """Encoder"""
    def train_encoder(self, obs, obs_pos, actions, obs_neg):
        feed_dict = {self.obs_pos: obs_pos,
                     self.obs: obs,
                     self.obs_neg: obs_neg,
                     self.actions: actions}

        _, loss = self.sess.run([self.op, self.loss], feed_dict = feed_dict)

        return loss

    def test_encoder(self, obs, obs_pos, actions, obs_neg, print_msg = None):
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()
        start = time.time()
        feed_dict = {self.obs_pos: obs_pos,
                     self.obs: obs,
                     self.obs_neg: obs_neg,
                     self.actions: actions}
        loss = self.sess.run([self.loss], feed_dict = feed_dict)[0]
        # loss = self.sess.run([self.loss], feed_dict = feed_dict, options=run_options, run_metadata=run_metadata)[0]
        if print_msg != None:
            print(print_msg, ":", loss)
        # tl = timeline.Timeline(run_metadata.step_stats)
        # ctf = tl.generate_chrome_trace_format()
        # with open('./timeline.json', 'w') as f:
        #     f.write(ctf)
        print("testing time:", time.time() - start)
        return loss

    def restore_encoder(self):
        scope = tf.get_default_graph().get_name_scope()
        if scope != '':
            scope = scope + '/'
        self.obs = self.sess.graph.get_tensor_by_name(scope + 'obs:0')
        self.obs_pos = self.sess.graph.get_tensor_by_name(scope + 'obs_pos:0')
        self.obs_neg = self.sess.graph.get_tensor_by_name(scope + 'obs_neg:0')
        self.actions = self.sess.graph.get_tensor_by_name(scope + 'actions:0')
        self.z = self.sess.graph.get_tensor_by_name(scope + 'z:0')
        self.loss = self.sess.graph.get_tensor_by_name(scope + 'loss:0')


    def save_model(self, model_dir, i = 999):
        saver = tf.train.Saver()
        saver.save(self.sess, model_dir, global_step = i)
        logger.log("saved successfully")


class Decoder:
    def __init__(self, cpc,
                       sess,
                       z_dim,
                       feature_dims,
                       fixed_num_of_contact,
                       contact_point_dim,
                       lr,
                       input_type = 'latent'):
        self.z_dim = z_dim
        self.feature_dims = feature_dims
        self.fixed_num_of_contact = fixed_num_of_contact
        self.contact_point_dim = contact_point_dim
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.cpc = cpc
        self.sess = sess

        self.out_dim = 3
        self.input_type = input_type
        self.object_pos = tf.placeholder(tf.float32, [None, self.out_dim], name = 'obs_pos')
        self.build_graph()


    def build_graph(self):
        zs = self.cpc.z
        recon = self.predict(zs, name = 'decoder_graph')
        self.loss = self.loss(self.object_pos, recon)
        self.loss = tf.identity(self.loss, name = 'decoder_loss')
        self.op = self.optimizer.minimize(self.loss)

    def predict(self, z, name):
        reuse = None
        x = z
        layers_sizes = self.feature_dims
        for i, size in enumerate(layers_sizes):
            activation = tf.nn.relu
            x = tf.layers.dense(inputs=x,
                                units=size,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                reuse=reuse,
                                name=name + '_' + str(i))
            x = activation(x)
        x = tf.layers.dense(inputs=x,
                            units=self.out_dim,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            reuse=reuse,
                            name=name + '_' + str(len(layers_sizes)))

        return x


    def loss(self, x, pred):
        """loss: including two parts, cross entropy for the one-hot part and MSE loss for position and force. """
        real_pos = self.object_pos
        pred_pos = pred

        loss = tf.losses.mean_squared_error(labels=real_pos, predictions=pred_pos, weights=0.5)
        return loss

    def train(self, obs, object_position):
        feed_dict = {self.object_pos: object_position,
                     self.cpc.obs: obs}
        _, loss = self.sess.run([self.op, self.loss], feed_dict = feed_dict)
        return loss

    def test(self, obs, object_position):
        feed_dict = {self.object_pos: object_position,
                     self.cpc.obs: obs}
        loss = self.sess.run([self.loss], feed_dict = feed_dict)[0]
        return loss

    def restore(self):
        scope = tf.get_default_graph().get_name_scope()
        if scope != '':
            scope = scope + '/'
        self.loss = self.sess.graph.get_tensor_by_name(scope + 'decoder_loss:0')
        self.object_pos = self.sess.graph.get_tensor_by_name(scope + 'object_pos:0')


    def save_model(self, model_dir, i = 999):
        saver = tf.train.Saver()
        saver.save(self.sess, model_dir, global_step = i)
        logger.log("saved successfully")
