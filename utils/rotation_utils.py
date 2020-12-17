import tensorflow as tf
from pdb import set_trace as st


def compute_rotation_matrix_from_ortho6d(ortho6d):
    with tf.variable_scope('6d_to_mat'):
        r6d = tf.reshape(ortho6d, [-1,6])
        x_raw = r6d[:,:3]
        y_raw = r6d[:,3:]

        x = tf.nn.l2_normalize(x_raw, axis=-1)
        z = tf.cross(x, y_raw)
        z = tf.nn.l2_normalize(z, axis=-1)
        y = tf.cross(z, x)

        x = tf.reshape(x, [-1,3,1])
        y = tf.reshape(y, [-1,3,1])
        z = tf.reshape(z, [-1,3,1])
        matrix = tf.concat([x,y,z], axis=-1)

    return matrix


def compute_geodesic_distance_from_two_matrices(m1, m2):
    with tf.variable_scope('error'):
        m2_new = tf.transpose(m2, perm=[0, 2, 1])
        m = tf.matmul(m1, m2_new)
        cos = tf.math.divide(m[:,0,0] + m[:,1,1] + m[:,2,2] - 1, 2)
        cos = tf.clip_by_value(cos, -0.999, 0.999)
        theta = tf.math.acos(cos)
    return tf.reduce_mean(theta)
