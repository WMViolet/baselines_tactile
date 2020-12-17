import tensorflow as tf
import numpy as np
from utils.utils import *
from pdb import set_trace as st
from tactile_baselines import logger
from scipy.spatial.transform import Rotation as R
from utils.mlp import *
from utils.rotation_utils import *

class RotationModel:
    def __init__(self, dims, sess, fixed_num_of_contact, feature_layers, output_layers, learning_rate):
        self.out_channel = 6
        self.sess = sess
        self.fixed_num_of_contact = fixed_num_of_contact
        self.feature_layers = feature_layers
        self.output_layers = output_layers
        self.learning_rate = learning_rate
        self.name = 'rotation'

        self.input = tf.placeholder(tf.float32, shape=[None, fixed_num_of_contact, dims[0]], name="input")
        self.labels = tf.placeholder(tf.float32, shape=[None, 3, 3], name="labels")
        self.build_graph()

    def build_graph(self):
        self.output = nn_max_pool(self.input, self.feature_layers, self.output_layers + [self.out_channel], name = self.name)
        self.predictions = compute_rotation_matrix_from_ortho6d(self.output) #batch*3*3
        self.predictions = tf.identity(self.predictions, name="output")
        self.geodesic_loss = compute_geodesic_distance_from_two_matrices(self.labels, self.predictions)
        self.geodesic_loss = tf.identity(self.geodesic_loss, name="loss")
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.op = optimizer.minimize(loss=self.geodesic_loss, var_list=var_list)

    def train(self, input_data, labels):
        feed_dict = {self.input: input_data,
                     self.labels: labels}
        loss, _ = self.sess.run([self.geodesic_loss, self.op], feed_dict=feed_dict)
        logger.logkv('train_classify_loss', loss)

    def test(self, input_data, labels):
        feed_dict = {self.input: input_data,
                     self.labels: labels}
        accuracy = self.sess.run([self.geodesic_loss], feed_dict=feed_dict)[0]
        logger.logkv('test_pred_loss', accuracy)

    def predict_single(self, input_data):
        feed_dict = {self.input: input_data}
        prediction = self.sess.run([self.predictions], feed_dict=feed_dict)[0]
        prediction = R.from_dcm(prediction).as_quat()
        return prediction


    def restore(self):
        self.input_ph = self.sess.graph.get_tensor_by_name('input:0')
        self.label_ph = self.sess.graph.get_tensor_by_name('labels:0')
        self.loss_ph = self.sess.graph.get_tensor_by_name('loss:0')
        self.output_ph = self.sess.graph.get_tensor_by_name('output:0')

    def restore_predict(self, inputs, labels):
        prediction, test_loss = self.sess.run([self.output_ph,
                                               self.loss_ph],
                                              feed_dict={self.input_ph: inputs,
                                                         self.label_ph: labels})
        logger.logkv('test_loss', test_loss)
        return prediction, test_loss

    def restore_predict_single(self, inputs):
        prediction = self.sess.run([self.output_ph],
                                    feed_dict={self.input_ph: inputs})[0].reshape((-1))
        return prediction

    def save_model(self, model_dir, i = 999):
        saver = tf.train.Saver()
        saver.save(self.sess, model_dir, global_step = i)
        print("rotation model is saved successfully")
