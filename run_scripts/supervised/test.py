import tensorflow as tf
import torch
import torch.nn as nn
import numpy as np
from pdb import set_trace as st
from scipy.spatial.transform import Rotation as R
from torch.autograd import Variable

sess = tf.InteractiveSession() 


quat1 = np.array([[0, 0, np.sin(np.pi/4), np.cos(np.pi/4)], [0, np.sin(np.pi/3), np.cos(np.pi/3), 0]])
quat2 = np.array([[np.sin(np.pi/5), np.cos(np.pi/5), 0, 0], [np.sin(np.pi/7), 0, 0, np.cos(np.pi/7)]])

matrix1 = R.from_quat(quat1).as_dcm().reshape((-1, 3, 3)).astype(np.float32)
matrix2 = R.from_quat(quat2).as_dcm().reshape((-1, 3, 3)).astype(np.float32)

pred6d = np.array([[1.,2.,3.,4.,5.,6.], [11.,23.,43.,48.,52.,36.]]).astype(np.float32)

tf_pred6d = tf.convert_to_tensor(pred6d)
torch_pred6d = torch.from_numpy(pred6d).float()

from utils.rotation_utils import *
tf_mat = compute_rotation_matrix_from_ortho6d(tf_pred6d)
# print(tf_mat.eval())

tf_loss = compute_geodesic_distance_from_two_matrices(tf.convert_to_tensor(matrix1), tf_mat)
print(tf.reduce_mean(tf_loss).eval())

print("****************************************************")
print("****************************************************")

from utils.torch_utils import *
torch_mat = compute_rotation_matrix_from_ortho6d(torch_pred6d)
# print(torch_mat)
torch_loss = compute_geodesic_distance_from_two_matrices(torch.from_numpy(matrix1).float(), torch_mat)
print(torch_loss.mean())

"""testing results recorded here:
   1. compute_rotation_matrix_from_ortho6d works.
   2. compute_geodesic_distance_from_two_matrices works.
   """
