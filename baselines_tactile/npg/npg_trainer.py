import tensorflow as tf
import numpy as np
import time
from tactile_baselines import logger

import abc
from collections import OrderedDict
from distutils.version import LooseVersion
from itertools import count
import math
import os
from pdb import set_trace as st
from tactile_baselines.common.replay_buffers import SimpleReplayBuffer
from pdb import set_trace as st


class Trainer(object):
    def __init__(
            self,
            algo,
            env,
            sampler,
            sample_processor,
            policy,
            n_itr,
            start_itr=0,
            task=None,
            sess=None,
            n_initial_exploration_steps=1e3,
            max_replay_buffer_size=int(1e6),
            epoch_length=1000,
            sample_batch_size=256,
            ):
        self.algo = algo
        self.env = env
        self.sampler = sampler
        self.sample_processor = sample_processor
        self.baseline = sample_processor.baseline
        self.policy = policy
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.task = task
        self.sample_batch_size = sample_batch_size
        self.n_initial_exploration_steps = n_initial_exploration_steps
        self.replay_buffer = SimpleReplayBuffer(self.env, max_replay_buffer_size)
        self.epoch_length = epoch_length
        self.num_grad_steps = self.sampler.total_samples
        if sess is None:
            sess = tf.Session()
        self.sess = sess

    def train(self):
        with self.sess.as_default() as sess:
            sess.run(tf.global_variables_initializer())
            start_time = time.time()

            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                logger.log("\n ---------------- Iteration %d ----------------" % itr)
                logger.log("Obtaining samples...")
                time_env_sampling_start = time.time()
                paths = self.sampler.obtain_samples(log=True, log_prefix='train-')
                sampling_time = time.time() - time_env_sampling_start

                """ ----------------- Processing Samples ---------------------"""
                # check how the samples are processed
                logger.log("Processing samples...")
                time_proc_samples_start = time.time()
                samples_data = self.sample_processor.process_samples(paths, log='all', log_prefix='train-')[0]
                self.replay_buffer.add_samples(samples_data['observations'],
                                               samples_data['actions'],
                                               samples_data['rewards'],
                                               samples_data['dones'],
                                               samples_data['next_observations'],
                                               )
                proc_samples_time = time.time() - time_proc_samples_start

                paths = self.sampler.obtain_samples(log=True, log_prefix='eval-', deterministic=True)
                _ = self.sample_processor.process_samples(paths, log='all', log_prefix='eval-')[0]

                """ ------------------ Policy Update ---------------------"""

                logger.log("Optimizing policy...")

                # This needs to take all samples_data so that it can construct graph for meta-optimization.
                time_optimization_step_start = time.time()

                samples_num = self.sample_batch_size

                samples = self.replay_buffer.random_batch_simple(samples_num)

                self.algo.optimize_policy(samples)

                """ ------------------- Logging Stuff --------------------------"""
                logger.logkv('Itr', itr)
                logger.logkv('n_timesteps', self.sampler.total_timesteps_sampled)
                logger.logkv('ItrTime', time.time() - itr_start_time)

                logger.dumpkvs()
                if itr == 0:
                    sess.graph.finalize()



                # Log information
                # if self.save_logs:
                #     self.logger.log_kv('alpha', alpha)
                #     self.logger.log_kv('delta', n_step_size)
                #     self.logger.log_kv('time_vpg', t_gLL)
                #     self.logger.log_kv('time_npg', t_FIM)
                #     self.logger.log_kv('kl_dist', kl_dist)
                #     self.logger.log_kv('surr_improvement', surr_after - surr_before)
                #     self.logger.log_kv('running_score', self.running_score)
                #     try:
                #         self.env.env.env.evaluate_success(paths, self.logger)
                #     except:
                #         # nested logic for backwards compatibility. TODO: clean this up.
                #         try:
                #             success_rate = self.env.env.env.evaluate_success(paths)
                #             self.logger.log_kv('success_rate', success_rate)
                #
