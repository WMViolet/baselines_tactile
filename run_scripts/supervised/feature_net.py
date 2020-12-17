# import tensorflow as tf
# from tactile_baselines import logger
# from pdb import set_trace as st
# from tactile_baselines.her.pointnet import get_features
# from utils.mlp import *
# import numpy as np
#
#
# def nn_max_pool(input, layers_sizes, reuse=None, name=""):
#     """Creates a simple neural network
#     """
#     for i, size in enumerate(layers_sizes):
#         activation = tf.nn.relu
#         input = tf.layers.dense(inputs=input,
#                                 units=size,
#                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                                 reuse=reuse,
#                                 name=name + '_' + str(i) + '/pre')
#         input = activation(input)
#     input = tf.reduce_max(input, axis = 1)
#     return input
#
# def nn(input, output_dims, reuse=None, name=""):
#     for i, size in enumerate(output_dims):
#         activation = tf.nn.relu if i < len(output_dims) - 1 else None
#         input = tf.layers.dense(inputs=input,
#                                 units=size,
#                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                                 reuse=reuse,
#                                 name=name + '_' + str(i) + '/remove')
#         if activation:
#             input = activation(input)
#     return input
#
#
# class FeatureNet:
#     def __init__(self,
#                  dims,
#                  fixed_num_of_contact,
#                  sess,
#                  process_type,
#                  position_layers,
#                  learning_rate):
#
#         self.sess = sess
#         self.input_dim = dims[0]
#         self.position_dim = dims[1]
#         self.input = tf.placeholder(tf.float32, shape=[None, fixed_num_of_contact, self.input_dim], name="input")
#         self.positions = tf.placeholder(tf.float32, shape=[None, self.position_dim], name="pos")
#
#         self.process_type = process_type
#         self.position_layers = position_layers
#         self.fixed_num_of_contact = fixed_num_of_contact
#         self.learning_rate = learning_rate
#
#         self.position_graph()
#
#     def get_features(self, contact):
#         features = nn_max_pool(contact, self.position_layers[0], name = 'max_pos')
#         return features
#
#     def position_graph(self):
#         name = 'position'
#         with tf.variable_scope(name):
#             if self.process_type == 'max_pool':
#                 features = self.get_features(self.input)
#                 self.predicted_pos = nn(features, self.position_layers[1] + [self.position_dim], name = 'pos')
#                 classify_loss = tf.losses.mean_squared_error(predictions=self.predicted_pos, labels=self.positions, weights=0.5)
#                 self.position_loss = self.position_cls_loss = classify_loss
#
#             var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
#             self.position_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='pos_optimizer')
#             self.pos_op = self.position_optimizer.minimize(loss=self.position_loss, var_list=var_list)
#
#         """ Not sure if they should be in the nane scope or not. """
#         self.predicted_pos = tf.identity(self.predicted_pos, name="predicted_pos")
#         self.position_loss = tf.identity(self.position_loss, name="position_loss")
#         self.features = tf.identity(features, name = 'features_tensor')
#
#
#     def train(self, input_data, position):
#         feed_dict = {self.input: input_data,
#                      self.positions: position}
#         position_loss, _ = self.sess.run([self.position_cls_loss,
#                                                             self.pos_op], feed_dict=feed_dict)
#         logger.logkv('train_position_loss', position_loss)
#
#     def test(self, input_data, position):
#         feed_dict = {self.input: input_data,
#                      self.positions: position}
#         position_loss = self.sess.run([self.position_cls_loss], feed_dict=feed_dict)[0]
#         logger.logkv('test_position_loss', position_loss)
#
#     def predict_single(self, input_data):
#         feed_dict = {self.input: input_data}
#         pos = self.sess.run([self.predicted_pos], feed_dict=feed_dict)
#         pos = pos.reshape((-1))
#         return np.concatenate(pos, axis = -1)
#
#     def restore(self):
#         self.input_ph = self.sess.graph.get_tensor_by_name('input:0')
#         self.positions_ph = self.sess.graph.get_tensor_by_name('pos:0')
#         self.position_loss_ph = self.sess.graph.get_tensor_by_name('position_loss:0')
#         self.predicted_pos_ph = self.sess.graph.get_tensor_by_name('predicted_pos:0')
#         self.features_tensor = self.sess.graph.get_tensor_by_name('features_tensor:0')
#         # self.test_ph = tf.placeholder(tf.float32, shape=[None, self.fixed_num_of_contact, self.input_dim], name="input")
#         # st()
#         # self.features1 = self.get_features(self.test_ph)
#
#     # def test_restore(self, inputs):
#     #     st()
#     #     feed_dict = {self.input_ph: inputs}
#     #     features2 = self.sess.run([self.features_ph], feed_dict = feed_dict)
#     #     feed_dict = {self.test_ph: inputs}
#     #     features1 = self.sess.run([self.features1], feed_dict = feed_dict)
#     #     features1, features2 = self.sess.run([features1, self.features_ph],
#     #                                           feed_dict={self.test_ph: inputs,
#     #                                                      self.input_ph: inputs})
#     #     st()
#     #     logger.logkv('restore_position_loss', position_loss)
#
#     def restore_predict(self, inputs, position):
#         pos, position_loss = self.sess.run([self.predicted_pos_ph, self.position_loss_ph],
#                                               feed_dict={self.input_ph: inputs,
#                                                          self.positions_ph: position})
#         logger.logkv('restore_position_loss', position_loss)
#
#
#     def restore_predict_single(self, inputs):
#         pos = self.sess.run([self.predicted_pos_ph],
#                                  feed_dict={self.input_ph: inputs})
#         return pos.reshape((-1))
#
#     def save_model(self, model_dir, i = 999):
#         saver = tf.train.Saver()
#         saver.save(self.sess, model_dir, global_step = i)
#         print("saved successfully")
