import tensorflow as tf
from tactile_baselines import logger
from pdb import set_trace as st
from tactile_baselines.her.pointnet import get_features
from scipy.spatial.transform import Rotation as R
from utils.rotation_utils import *
from utils.mlp import *
import numpy as np


class FeatureNet:
    def __init__(self,
                 dims,
                 fixed_num_of_contact,
                 sess,
                 process_type,
                 position_layers,
                 rotation_layers,
                 learning_rates):

        self.sess = sess
        self.input_dim = dims[0]
        self.position_dim = dims[1]
        self.rotation_dim = 6
        self.input = tf.placeholder(tf.float32, shape=[None, fixed_num_of_contact, self.input_dim], name="input")
        self.rotations = tf.placeholder(tf.float32, shape=[None, 3, 3], name="rot")
        self.positions = tf.placeholder(tf.float32, shape=[None, self.position_dim], name="pos")

        self.process_type = process_type
        self.position_layers = position_layers
        self.rotation_layers = rotation_layers
        self.fixed_num_of_contact = fixed_num_of_contact
        self.learning_rates = learning_rates

        self.position_graph()
        self.rotation_graph()


    def position_graph(self):
        name = 'position'
        with tf.variable_scope(name):
            if self.process_type == 'pointnet':
                self.predicted_pos, transform = get_features(self.input, self.position_dim)
                classify_loss = tf.losses.mean_squared_error(predictions=self.predicted_pos, labels=self.positions, weights=0.5)
                mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
                mat_diff -= tf.constant(np.eye(transform.get_shape()[1].value), dtype=tf.float32)
                mat_diff_loss = tf.nn.l2_loss(mat_diff)
                reg_weight=0.001
                self.position_loss = classify_loss + mat_diff_loss * reg_weight
                self.position_cls_loss = classify_loss
            if self.process_type == 'max_pool':
                self.predicted_pos = nn_max_pool(self.input, self.position_layers[0], self.position_layers[1] + [self.position_dim])
                classify_loss = tf.losses.mean_squared_error(predictions=self.predicted_pos, labels=self.positions, weights=0.5)
                self.position_loss = self.position_cls_loss = classify_loss

            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
            self.position_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rates[0], name='pos_optimizer')
            self.pos_op = self.position_optimizer.minimize(loss=self.position_loss, var_list=var_list)

        """ Not sure if they should be in the nane scope or not. """
        self.predicted_pos = tf.identity(self.predicted_pos, name="predicted_pos")
        self.position_loss = tf.identity(self.position_loss, name="position_loss")


    def rotation_graph(self):
        name = 'rotation'
        with tf.variable_scope(name):
            if self.process_type == 'pointnet':
                features, transform = get_features(self.input, self.rotation_dim)
                self.predicted_rot = compute_rotation_matrix_from_ortho6d(features)
                classify_loss = compute_geodesic_distance_from_two_matrices(self.predicted_rot, self.rotations)
                mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
                mat_diff -= tf.constant(np.eye(transform.get_shape()[1].value), dtype=tf.float32)
                mat_diff_loss = tf.nn.l2_loss(mat_diff)
                reg_weight=0.001
                self.rotation_loss = classify_loss + mat_diff_loss * reg_weight
                self.rotation_cls_loss = classify_loss
            if self.process_type == 'max_pool':
                features = nn_max_pool(self.input, self.rotation_layers[0], self.rotation_layers[1] + [self.rotation_dim])
                self.predicted_rot = compute_rotation_matrix_from_ortho6d(features)
                self.rotation_loss = self.rotation_cls_loss = compute_geodesic_distance_from_two_matrices(self.predicted_rot, self.rotations)

            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
            self.rotation_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rates[1], name='rot_optimizer')
            self.rot_op = self.rotation_optimizer.minimize(loss=self.rotation_loss, var_list=var_list)

        """ Not sure if they should be in the nane scope or not. """
        self.predicted_rot = tf.identity(self.predicted_rot, name="predicted_rot")
        self.rotation_loss = tf.identity(self.rotation_loss, name="rotation_loss")


    def train(self, input_data, position, rot_matrix):
        feed_dict = {self.input: input_data,
                     self.rotations: rot_matrix,  #batch*3*3
                     self.positions: position}
        position_loss, _, rotation_loss, _ = self.sess.run([self.position_cls_loss,
                                                            self.pos_op,
                                                            self.rotation_cls_loss,
                                                            self.rot_op], feed_dict=feed_dict)
        logger.logkv('train_position_loss', position_loss)
        logger.logkv('train_rotation_loss', rotation_loss)

    def test(self, input_data, position, rot_matrix):
        feed_dict = {self.input: input_data,
                     self.rotations: rot_matrix,  #batch*3*3
                     self.positions: position}
        position_loss, rotation_loss = self.sess.run([self.position_cls_loss, self.rotation_cls_loss], feed_dict=feed_dict)
        logger.logkv('test_position_loss', position_loss)
        logger.logkv('test_rotation_loss', rotation_loss)

    def predict_single(self, input_data):
        feed_dict = {self.input: input_data}
        pos, rot = self.sess.run([self.predicted_pos, self.predicted_rot], feed_dict=feed_dict)
        rot = R.from_dcm(rot).as_quat()
        pos, rot = pos.reshape((-1)), rot.reshape((-1))
        return np.concatenate((pos, rot), axis = -1)

    def restore(self):
        self.input_ph = self.sess.graph.get_tensor_by_name('input:0')
        self.rotations_ph = self.sess.graph.get_tensor_by_name('rot:0')
        self.positions_ph = self.sess.graph.get_tensor_by_name('pos:0')
        self.rotation_loss_ph = self.sess.graph.get_tensor_by_name('rotation_loss:0')
        self.position_loss_ph = self.sess.graph.get_tensor_by_name('position_loss:0')
        self.predicted_rot_ph = self.sess.graph.get_tensor_by_name('predicted_rot:0')
        self.predicted_pos_ph = self.sess.graph.get_tensor_by_name('predicted_pos:0')

    def restore_predict(self, inputs, position, rot_matrix):
        pos, rot, position_loss, rotation_loss = self.sess.run([self.predicted_pos_ph,
                                                                self.predicted_rot_ph,
                                                                self.position_loss_ph,
                                                                self.rotation_loss_ph],
                                              feed_dict={self.input_ph: inputs,
                                                         self.rotations_ph: rot_matrix,  #batch*3*3
                                                         self.positions_ph: position})
        logger.logkv('restore_position_loss', position_loss)
        logger.logkv('restore_rotation_loss', rotation_loss)


    def restore_predict_single(self, inputs):
        pos, rot = self.sess.run([self.predicted_pos_ph, self.predicted_rot_ph],
                                 feed_dict={self.input_ph: inputs})
        rot = R.from_dcm(rot).as_quat()
        pos, rot = pos.reshape((-1)), rot.reshape((-1))
        return np.concatenate((pos, rot), axis = -1)

    def save_model(self, model_dir, i = 999):
        saver = tf.train.Saver()
        saver.save(self.sess, model_dir, global_step = i)
        print("saved successfully")
