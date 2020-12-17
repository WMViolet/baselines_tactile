import tensorflow as tf
import numpy as np
import math
import sys
import os
from tactile_baselines.her.tf_util import *
from tactile_baselines.her.transform_nets import input_transform_net, feature_transform_net
from pdb import set_trace as st

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl


# def get_model(point_cloud, is_training=tf.constant(True), bn_decay=None):
def get_features(point_cloud, out_dim):
    is_training=tf.constant(True)
    bn_decay=None
    """ Classification PointNet, input is BxNxinput_dim, output Bx40 """
    batch_size = tf.shape(point_cloud)[0]
    num_point = point_cloud.get_shape()[1].value
    input_dim = point_cloud.get_shape()[2].value
    end_points = {}

    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay)
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1)

    net = conv2d(input_image, 32, [1,input_dim],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = conv2d(net, 32, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)

    with tf.variable_scope('transform_net2') as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=32)
    end_points['transform'] = transform
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
    net_transformed = tf.expand_dims(net_transformed, [2])

    net = conv2d(net_transformed, 32, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)
    # Symmetric function: max pooling
    net = max_pool2d(net, [num_point,1],
                             padding='VALID', scope='maxpool')

    net = tf.reshape(net, [-1, net.shape[2]*net.shape[3]*net.shape[1]])
    net = fully_connected(net, 64, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = fully_connected(net, 32, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp2')
    net = fully_connected(net, out_dim, activation_fn=None, scope='fc3')

    return net, transform
