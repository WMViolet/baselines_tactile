import tensorflow as tf
from tactile_baselines import logger
from pdb import set_trace as st
from tactile_baselines.her.pointnet import get_features
from utils.mlp import *
import numpy as np


def supervised_nn_max_pool(input, layers_sizes, output_dims, reuse=None, name=""):
    """Creates a simple neural network
    """
    # input = tf.reshape(input, (-1, np.prod(input.shape[1:])))
    for i, size in enumerate(layers_sizes):
        activation = tf.nn.relu
         # if i < len(output_dims) - 1 else tf.math.tanh
        input = tf.layers.dense(inputs=input,
                                units=size,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                reuse=reuse,
                                name=name + 'pre_train_' + str(i))
        input = activation(input)

    input = tf.reduce_mean(input, axis = 1)

    for i, size in enumerate(output_dims):
        activation = tf.nn.relu if i < len(output_dims) - 1 else None
        input = tf.layers.dense(inputs=input,
                                units=size,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                reuse=reuse,
                                name=name + 'remove_' + str(i+len(layers_sizes)))
        if activation:
            input = activation(input)
    # trainable vairables:
    # tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='')
    # <tf.Variable 'position/pre_train_0/kernel:0' shape=(3, 32) dtype=float32_ref>,
    # <tf.Variable 'position/pre_train_0/bias:0' shape=(32,) dtype=float32_ref>,
    # <tf.Variable 'position/pre_train_1/kernel:0' shape=(32, 32) dtype=float32_ref>,
    # <tf.Variable 'position/pre_train_1/bias:0' shape=(32,) dtype=float32_ref>
    # <tf.Variable 'position/remove_2/kernel:0' shape=(32, 32) dtype=float32_ref>
    # <tf.Variable 'position/remove_2/bias:0' shape=(32,) dtype=float32_ref>
    # <tf.Variable 'position/remove_3/kernel:0' shape=(32, 32) dtype=float32_ref>
    # <tf.Variable 'position/remove_3/bias:0' shape=(32,) dtype=float32_ref>
    # <tf.Variable 'position/remove_4/kernel:0' shape=(32, 3) dtype=float32_ref>
    # <tf.Variable 'position/remove_4/bias:0' shape=(3,) dtype=float32_ref>
    return input



class FeatureNet:
    def __init__(self,
                 dims,
                 fixed_num_of_contact,
                 sess,
                 process_type,
                 position_layers,
                 learning_rate):

        self.sess = sess
        self.input_dim = dims[0]
        self.position_dim = dims[1]
        self.fixed_num_of_contact = fixed_num_of_contact
        self.input = tf.placeholder(tf.float32, shape=[None, fixed_num_of_contact, self.input_dim], name="input")
        self.positions = tf.placeholder(tf.float32, shape=[None, self.position_dim], name="pos")

        self.process_type = process_type
        self.position_layers = position_layers
        self.fixed_num_of_contact = fixed_num_of_contact
        self.learning_rate = learning_rate

        self.position_graph()



    def position_graph(self):
        name = 'position'
        with tf.variable_scope(name):
            if self.process_type == 'max_pool':
                self.predicted_pos = supervised_nn_max_pool(self.input, self.position_layers[0], self.position_layers[1] + [self.position_dim])
                # self.position_loss = tf.losses.mean_squared_error(predictions=self.predicted_pos, labels=self.positions, weights=0.5)
                self.position_loss = tf.reduce_mean((self.predicted_pos - self.positions) ** 2)

            elif self.process_type == 'none':
                self.predicted_pos = feature_nn(self.input, self.position_layers[1] + [self.position_dim])

            self.position_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='pos_optimizer')
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_default_graph().get_name_scope())
            self.pos_op = self.position_optimizer.minimize(loss=self.position_loss, var_list=var_list)

        self.predicted_pos = tf.identity(self.predicted_pos, name="predicted_pos")
        self.position_loss = tf.identity(self.position_loss, name="position_loss")


    def train(self, input_data, position):
        feed_dict = {self.input: input_data,
                     self.positions: position}

        for _ in range(20):
            position_loss, _ = self.sess.run([self.position_loss,
                                             self.pos_op], feed_dict=feed_dict)
        # predictions = self.sess.run([self.predicted_pos], feed_dict=feed_dict)[0]
        logger.logkv('train_position_loss', position_loss)


    def test(self, input_data, position, log_info = '', print_msg = False):
        feed_dict = {self.input: input_data,
                     self.positions: position}
        position_loss = self.sess.run([self.position_loss], feed_dict=feed_dict)[0]
        logger.logkv(log_info + 'test_position_loss', position_loss)
        if print_msg:
            print(log_info + 'test_position_loss', position_loss)


    def restore(self):
        scope = tf.get_default_graph().get_name_scope()
        if scope != '':
            scope = scope + '/'

        self.input = self.sess.graph.get_tensor_by_name(scope + 'input:0')
        self.positions = self.sess.graph.get_tensor_by_name(scope + 'pos:0')
        self.position_loss = self.sess.graph.get_tensor_by_name(scope + 'position_loss:0')
        self.predicted = self.sess.graph.get_tensor_by_name(scope + 'predicted_pos:0')
        return scope + 'input:0'



    def save_model(self, model_dir, i = 999):
        saver = tf.train.Saver()
        saver.save(self.sess, model_dir, global_step = i)
        print("saved successfully")
