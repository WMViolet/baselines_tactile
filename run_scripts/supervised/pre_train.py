# from envs import gym
# from collections import defaultdict
# import os
# import json
# import tensorflow as tf
# import numpy as np
# import time
#
# from supervised.feature_net import FeatureNet
# from tactile_baselines import logger
# from tactile_baselines.utils.utils import set_seed, ClassEncoder
# from pdb import set_trace as st
# from utils.utils import *
# import dill as pickle
# import numpy as np
#
#
# def split_data(paths, niter):
#     episode_size = paths['positions'].shape[0]
#     index = np.arange(episode_size)
#     np.random.shuffle(index)
#     train_index = index[:int(0.8 * episode_size)]
#     test_index = index[int(0.8 * episode_size):]
#     train_dict, test_dict = {}, {}
#     for key in paths.keys():
#         data = paths[key]
#         train_data, test_data = data[train_index], data[test_index]
#         if len(train_data.shape) == 2:
#             train_dict[key] = train_data.reshape((niter, -1, train_data.shape[1]))
#             test_dict[key] = test_data.reshape((niter, -1, train_data.shape[1]))
#         elif len(train_data.shape) == 3:
#             train_dict[key] = train_data.reshape((niter, -1, train_data.shape[1], train_data.shape[2]))
#             test_dict[key] = test_data.reshape((niter, -1, train_data.shape[1], train_data.shape[2]))
#
#     return train_dict, test_dict
#
#
#
# def main(**kwargs):
#     exp_dir = os.getcwd() + '/save_model/' + kwargs['process_type'][0] + '/'
#     logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last')
#     json.dump(kwargs, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     config.gpu_options.per_process_gpu_memory_fraction = kwargs.get('gpu_frac', 0.95)
#     sess = tf.Session(config=config)
#
#     mode = kwargs['mode'][0]
#     visualize_training_data = kwargs['visualize_training_data'][0]
#     visualize_testing_data = kwargs['visualize_testing_data'][0]
#     batch_size = kwargs['batch_size'][0]
#     site_mode = kwargs['site_mode'][0]
#     input_label = kwargs['input_label'][0]
#     output_label = kwargs['output_label'][0]
#     time_interval = kwargs['time_interval'][0]
#
#     if mode == 'restore':
#         saver = tf.train.import_meta_graph(exp_dir + '-999.meta')
#         saver.restore(sess, tf.train.latest_checkpoint(exp_dir))
#         graph = tf.get_default_graph()
#
#     with sess.as_default() as sess:
#         buffer, fixed_num_of_contact = pickle.load(open('../saved/HandManipulateEgg-v0-fix9.pickle', 'rb'))
#
#         # Only to get the environment parameters
#         env = gym.make(kwargs['env'][0],
#                        obs_type = kwargs['obs_type'][0],
#                        fixed_num_of_contact = [fixed_num_of_contact, False])
#
#         ngeoms = env.sim.model.ngeom
#         paths = data_filter(buffer, fixed_num_of_contact, batch_size)
#         paths = expand_data(paths, ngeoms, fixed_num_of_contact)
#         niters = paths['positions'].shape[0] // batch_size
#         print("total iteration: ", niters)
#         print("total number of data: ", paths['positions'].shape[0])
#
#         train_data, test_data = split_data(paths, niters)
#         train_data['object_position'] = train_data['object_position'][:, :, :3]
#         # vis_data['original_object_position'] = vis_data['object_position']
#         # vis_data_test['original_object_position'] = vis_data_test['object_position']
#         test_data['object_position'] = test_data['object_position'][:, :, :3]
#
#         labels_to_dims = {}
#         labels_to_dims['positions'] = 3
#         labels_to_dims['object_position'] = 3
#         labels_to_dims['joint_position'] = 24
#         labels_to_dims['geoms'] = ngeoms
#
#         learning_rate = kwargs['learning_rate'][0]
#         position_layers = kwargs['position_layers'][0]
#         process_type = kwargs['process_type'][0]
#
#
#
#         dims = (labels_to_dims[input_label], labels_to_dims[output_label])
#
#         feature_net = FeatureNet(dims,
#                                  fixed_num_of_contact = fixed_num_of_contact,
#                                  sess = sess,
#                                  process_type = process_type,
#                                  position_layers = position_layers,
#                                  learning_rate = learning_rate)
#
#         if mode == 'train':
#             sess.run(tf.global_variables_initializer())
#             logger.log("training started")
#             for i in range(niters):
#                 start = time.time()
#                 feature_net.train(train_data[input_label][i], train_data[output_label][i])
#                 feature_net.test(test_data[input_label][i], test_data[output_label][i])
#                 logger.logkv("iter", i)
#                 logger.dumpkvs()
#             feature_net.save_model(exp_dir, 999)
#
#         if mode == 'restore':
#             feature_net.restore()
#             for i in range(1):
#                 logger.logkv("iter", i)
#                 feature_net.restore_predict(train_data[input_label][i], train_data[output_label][i])
#
#                 with open(exp_dir + 'params.pickle', 'wb') as pickle_file:
#                     pickle.dump([fixed_num_of_contact, dims, position_layers, learning_rate], pickle_file)
#
#                 with open(exp_dir + 'data.pickle', 'wb') as pickle_file:
#                     pickle.dump([train_data[input_label][1], train_data[output_label][1]], pickle_file)
#                 logger.dumpkvs()
#
#
#
#
#
# if __name__ == '__main__':
#     sweep_params = {
#         'alg': ['her'],
#         'seed': [1],
#         'env': ['HandManipulateEgg-v0'],
#
#         # Env Sampling
#         'num_timesteps': [1e6],
#         'buffer_size': [1e6],
#
#         # Problem Confs
#         'obs_type': ['full_contact'],
#         'process_type': ['max_pool'],
#         'input_label': ['positions'],
#         'output_label': ['object_position'],
#         'position_layers': [[[32, 32], [32, 32]]],
#         'learning_rate': [1e-3],
#         'mode': ['restore'],
#         'visualize_training_data': [False],
#         'visualize_testing_data': [False],
#         'batch_size': [100],
#         'time_interval': [1],
#         'site_mode': ['concurrent'],
#         }
#     main(**sweep_params)
