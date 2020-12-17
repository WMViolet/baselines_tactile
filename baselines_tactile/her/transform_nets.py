import tensorflow as tf
import numpy as np
from tactile_baselines.her.tf_util import *
from pdb import set_trace as st
from tactile_baselines.her.util import store_args, nn

def input_transform_net(point_cloud, is_training, bn_decay=None):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return:
            Transformation matrix of size 3xK """
    batch_size = tf.shape(point_cloud)[0]
    num_point = point_cloud.get_shape()[1].value
    K = point_cloud.get_shape()[2].value

    input_image = tf.expand_dims(point_cloud, -1)
    net = conv2d(input_image, 32, [1,K],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    net = conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay)
    # net = conv2d(net, 1024, [1,1],
    #                      padding='VALID', stride=[1,1],
    #                      bn=True, is_training=is_training,
    #                      scope='tconv3', bn_decay=bn_decay)
    net = max_pool2d(net, [num_point,1],
                             padding='VALID', scope='tmaxpool')
    # TODO: check the dimension here
    net = tf.reshape(net, [-1, 64])
    # net = nn(net, [512, 256])
    # net = fully_connected(net, 512, bn=True, is_training=is_training,
    #                               scope='tfc1', bn_decay=bn_decay)
    net = fully_connected(net, 64, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_XYZ') as sc:
        # assert(K==3)
        weights = tf.get_variable('weights', [64, K*K],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [K*K],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        identity = tf.reshape(tf.eye(K), [-1])
        # biases += tf.constant([1,0,0,0,1,0,0,0,1], dtype=tf.float32)
        biases += identity
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [-1, K, K])
    return transform


def feature_transform_net(inputs, is_training, bn_decay=None, K=64):
    """ Feature Transform Net, input is BxNx1xK
        Return:
            Transformation matrix of size KxK """
    batch_size = tf.shape(inputs)[0]
    num_point = inputs.get_shape()[1].value

    net = conv2d(inputs, 16, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    net = conv2d(net, 32, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay)
    net = conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay)
    net = max_pool2d(net, [num_point,1],
                             padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [-1, net.shape[2]*net.shape[3]*net.shape[1]])
    net = fully_connected(net, 64, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = fully_connected(net, 32, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_feat') as sc:
        weights = tf.get_variable('weights', [32, K*K],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [K*K],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        biases += tf.constant(np.eye(K).flatten(), dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, K, K])
    return transform
