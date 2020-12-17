import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
from baselines.her.pointnet2.utils.tf_util import *
from pointnet_util import pointnet_sa_module, pointnet_sa_module_msg

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl


def get_features(point_cloud, out_dim):
    is_training = tf.constant(True)
    bn_decay = None
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = tf.shape(point_cloud)[0]
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    l0_xyz = point_cloud
    l0_points = None

    # Set abstraction layers
    l1_xyz, l1_points = pointnet_sa_module_msg(l0_xyz, l0_points, 4, [0.1,0.2,0.4], [4,4,4], [[16,16,32], [16,16,32], [32,64,64]], is_training, bn_decay, scope='layer1', use_nchw=True)
    l2_xyz, l2_points = pointnet_sa_module_msg(l1_xyz, l1_points, 2, [0.2,0.4,0.8], [2,2,2], [[8,8,16], [16,16,32], [16,32,32]], is_training, bn_decay, scope='layer2')
    l3_xyz, l3_points, _ = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[32,32,64], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3')

    # Fully connected layers
    net = tf.reshape(l3_points, [-1, l3_points.shape[1] * l3_points.shape[2]])
    # net = fully_connected(net, 64, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    # net = dropout(net, keep_prob=0.4, is_training=is_training, scope='dp1')
    # net = fully_connected(net, 64, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    # net = dropout(net, keep_prob=0.4, is_training=is_training, scope='dp2')
    net = fully_connected(net, out_dim, activation_fn=None, scope='fc3')

    return net
