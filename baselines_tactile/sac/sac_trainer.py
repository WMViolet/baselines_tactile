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
    """
    Performs steps for MAML
    Args:
        algo (Algo) :
        env (Env) :
        sampler (Sampler) :
        sample_processor (SampleProcessor) :
        baseline (Baseline) :
        policy (Policy) :
        n_itr (int) : Number of iterations to train for
        start_itr (int) : Number of iterations policy has already trained for, if reloading
        num_inner_grad_steps (int) : Number of inner steps per maml iteration
        sess (tf.Session) : current tf session (if we loaded policy, for example)
    """
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
        self.n_initial_exploration_steps = n_initial_exploration_steps
        self.replay_buffer = SimpleReplayBuffer(self.env, max_replay_buffer_size)
        self.epoch_length = epoch_length
        self.num_grad_steps = self.sampler.total_samples
        if sess is None:
            sess = tf.Session()
        self.sess = sess

    def train(self):
        """
        Trains policy on env using algo
        Pseudocode:
            for itr in n_itr:
                for step in num_inner_grad_steps:
                    sampler.sample()
                    algo.compute_updated_dists()
                algo.optimize_policy()
                sampler.update_goals()
        """

        with self.sess.as_default() as sess:
            # initialize uninitialized vars  (only initialize vars that were not loaded)
            sess.run(tf.global_variables_initializer())
            start_time = time.time()

            if self.start_itr == 0:
                self.algo._update_target(tau=1.0)
                if self.n_initial_exploration_steps > 0:
                    while self.replay_buffer._size < self.n_initial_exploration_steps:
                        paths = self.sampler.obtain_samples(log=True, log_prefix='train-', random=True)
                        samples_data = self.sample_processor.process_samples(paths, log='all', log_prefix='train-')[0]
                        self.replay_buffer.add_samples(samples_data['observations'],
                                                       samples_data['actions'],
                                                       samples_data['rewards'],
                                                       samples_data['dones'],
                                                       samples_data['next_observations'],
                                                       )

            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                logger.log("\n ---------------- Iteration %d ----------------" % itr)
                logger.log("Sampling set of tasks/goals for this meta-batch...")

                """ -------------------- Sampling --------------------------"""

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

                # self.log_diagnostics(paths, prefix='train-')

                """ ------------------ Policy Update ---------------------"""

                logger.log("Optimizing policy...")

                # This needs to take all samples_data so that it can construct graph for meta-optimization.
                time_optimization_step_start = time.time()

                self.algo.optimize_policy(self.replay_buffer, itr * self.epoch_length, self.num_grad_steps)

                """ ------------------- Logging Stuff --------------------------"""
                logger.logkv('Itr', itr)
                logger.logkv('n_timesteps', self.sampler.total_timesteps_sampled)

                logger.logkv('Time-Optimization', time.time() - time_optimization_step_start)
                logger.logkv('Time-SampleProc', np.sum(proc_samples_time))
                logger.logkv('Time-Sampling', sampling_time)

                logger.logkv('Time', time.time() - start_time)
                logger.logkv('ItrTime', time.time() - itr_start_time)

                logger.dumpkvs()
                if itr == 0:
                    sess.graph.finalize()

        logger.log("Training finished")
        self.sess.close()

    def get_itr_snapshot(self, itr):
        """
        Gets the current policy and env for storage
        """
        return dict(itr=itr, policy=self.policy, env=self.env, baseline=self.baseline)
