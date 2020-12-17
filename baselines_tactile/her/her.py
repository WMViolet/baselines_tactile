import os

import click
import numpy as np
import json
from mpi4py import MPI
import time
import tactile_baselines.common.tf_util as U
from tactile_baselines.her.supervised.model import FeatureNet

from tactile_baselines import logger
from tactile_baselines.common import set_global_seeds, tf_util
from tactile_baselines.common.mpi_moments import mpi_moments
import tactile_baselines.her.experiment.config as config
from tactile_baselines.her.rollout import RolloutWorker
from tactile_baselines.common.mpi_fork import mpi_fork
from pdb import set_trace as st
from tactile_baselines.her.normalizer import Normalizer
import tensorflow as tf
import dill as pickle
import wandb

def mpi_average(value):
    if not isinstance(value, list):
        value = [value]
    if not any(value):
        value = [0.]
    return mpi_moments(np.array(value))[0]


def train(*, policy, rollout_worker, evaluator,
          n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval,
          save_path, demo_file, exp_dir, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()


    logger.info("Training...")
    best_success_rate = -1

    # num_timesteps = n_epochs * n_cycles * rollout_length * number of rollout workers
    if policy.pre_train_model == 'supervised':
        # test_input, test_output = pickle.load(open(policy.feature_net_path + 'data.pickle', 'rb'))
        stored_weghts = pickle.load(open(policy.feature_net_path + 'weights.pickle', 'rb'))
        restored_weights = [tf.constant(w) for w in stored_weghts]
        """assign weights for main"""
        new_scope_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ddpg/main/pi/process/predicted_pos')
        update_weights_main = [tf.assign(new, old) for (new, old) in zip(new_scope_main, restored_weights)]
        policy.sess.run(update_weights_main)
        """assign weights for target"""
        new_scope_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ddpg/target/pi/process/predicted_pos')
        update_weights_target = [tf.assign(new, old) for (new, old) in zip(new_scope_target, restored_weights)]
        policy.sess.run(update_weights_target)
    elif policy.pre_train_model == 'cpc':
        path = '/home/vioichigo/try/tactile-baselines/saved_model/cpc2/max_pool/trained/'
        stored_weights = pickle.load(open(path + 'weights.pickle', 'rb'))
        restored_weights = [tf.constant(w) for w in stored_weights]
        """assign weights for main"""
        new_scope_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ddpg/main/pi/process/new_cpc')
        update_weights_main = [tf.assign(new, old) for (new, old) in zip(new_scope_main, restored_weights)]
        policy.sess.run(update_weights_main)
        """assign weights for target"""
        new_scope_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ddpg/target/pi/process/new_cpc')
        update_weights_target = [tf.assign(new, old) for (new, old) in zip(new_scope_target, restored_weights)]
        policy.sess.run(update_weights_target)




    for epoch in range(n_epochs): #200

        if policy.pre_train_model != 'none':
            auxiliary_loss = []
        start_time = time.time()
        # train
        rollout_worker.clear_history()
        for n_cycle in range(n_cycles): #50
            episode = rollout_worker.generate_rollouts()
            obs = policy.store_episode(episode)
            start_here = time.time()

            for i in range(n_batches): #40
                if policy.pre_train_model == 'none':
                    policy.train()
                else:
                    _, _, loss = policy.train()
                    auxiliary_loss.append(loss)
            policy.update_target_net()

        # test
        evaluator.clear_history()
        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts()

        # record logs
        logger.record_tabular('epoch', epoch)
        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))

        logger.logkv('itr time', time.time() - start_time)

        if policy.pre_train_model == 'supervised':
            logger.logkv('auxiliary loss', np.array(auxiliary_loss).mean())

        if rank == 0:
            log_dict = dict([])
            for k in logger.Logger.CURRENT.name2val:
                value = logger.Logger.CURRENT.name2val[k]
                log_dict[k] = np.mean([value])
            wandb.log(log_dict)

        if rank == 0:
            logger.dump_tabular()

        # save the policy if it's better than the previous ones
        success_rate = mpi_average(evaluator.current_success_rate())
        # can't pickle SwigPyObject objects
        # if rank == 0 and success_rate >= best_success_rate and save_path and epoch % 10 == 0:
        #     best_success_rate = success_rate
            # batch = policy.sample_batch()
            # policy.sess.run(policy.stage_op, feed_dict=dict(zip(policy.buffer_ph_tf, batch)))
            # critic_loss, actor_loss, Q_grad, pi_grad = policy._grads()
            # print("calculated")
            # if policy.pre_train_model == 'supervised':
            #     policy.sess.run(policy.stage_op, feed_dict=dict(zip(policy.buffer_ph_tf, batch)))
            #     feature_loss, feature_grad = policy.sess.run([policy.feature_loss_tf, policy.feature_grad_tf])
            #     with open(save_path + '/stats.pickle', 'wb') as pickle_file:
            #         pickle.dump([batch, critic_loss, actor_loss, Q_grad, pi_grad, feature_loss, feature_grad], pickle_file)
            # else:
            #     with open(save_path + '/stats.pickle', 'wb') as pickle_file:
            #         pickle.dump([batch, critic_loss, actor_loss, Q_grad, pi_grad], pickle_file)

            # policy.o_stats.save(save_path + '/o-stats' + str(epoch) + '.pickle')
            # if policy.pre_train_model == 'cpc':
            #     policy.feature_stats.save(save_path + '/feature-stats' + str(epoch) + '.pickle')
            # print("model saved")

            # actually includes the two steps above
            # tf_util.save_variables(save_path + '/saved' + str(epoch) + '.pkl', sess=policy.sess)
        if save_path and success_rate >= best_success_rate and epoch % 10 == 0:
            best_success_rate = success_rate
            tf_util.save_variables(save_path + '/saved' + str(epoch) + '-seed' + str(rank) + '.pkl', sess=policy.sess)
            # print("vars saved")




        policy.sess.run(policy.increment_global_step)
        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)


        if rank != 0:
            assert local_uniform[0] != root_uniform[0]



    return policy





def learn(*, network, env, total_timesteps, num_cpu,
    seed=None,
    eval_env=None,
    replay_strategy='future',
    policy_save_interval=5,
    clip_return=True,
    demo_file=None,
    override_params=None,
    load_path=None,
    save_path=None,
    obs_type='original',
    process_type='none',
    feature_dim=64,
    fixed_num_of_contact=0,
    pre_train_model=False,
    include_action=False,
    update_model=True,
    feature_net_path='',
    **kwargs
):

    override_params = override_params or {}
    if MPI is not None:
        rank = MPI.COMM_WORLD.Get_rank()
    if num_cpu > 1:
        whoami = mpi_fork(num_cpu)
        if whoami == 'parent':
            sys.exit(0)

        U.single_threaded_session().__enter__()


    # Seed everything.
    rank_seed = seed + 1000000 * rank if seed is not None else None
    set_global_seeds(rank_seed)

    # Prepare params.
    params = config.DEFAULT_PARAMS
    env_name = env.spec.id
    params['env_name'] = env_name
    params['replay_strategy'] = replay_strategy
    params['obs_type'] = obs_type
    params['fixed_num_of_contact'] = fixed_num_of_contact
    params['process_type'] = process_type
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params.update(**override_params)  # makes it possible to override any parameter
    params = config.prepare_params(params)
    params['rollout_batch_size'] = env.num_envs

    if demo_file is not None:
        params['bc_loss'] = 1
    params.update(kwargs)

    config.log_params(params, logger=logger)
    exp_dir = kwargs['path']

    if rank == 0:
        relevant_params = {}
        for key in params:
            if key not in ['_network_class', 'network_class', 'make_env']:
                relevant_params[key] = params[key]
        wandb.init(project=params['proj_name'], config=relevant_params)

    if num_cpu == 1:
        logger.warn()
        logger.warn('*** Warning ***')
        logger.warn(
            'You are running HER with just a single MPI worker. This will work, but the ' +
            'experiments that we report in Plappert et al. (2018, https://arxiv.org/abs/1802.09464) ' +
            'were obtained with --num_cpu 19. This makes a significant difference and if you ' +
            'are looking to reproduce those results, be aware of this. Please also refer to ' +
            'https://github.com/openai/baselines/issues/314 for further details.')
        logger.warn('****************')
        logger.warn()

    dims = config.configure_dims(params)
    contact_dim = env.envs[0].env.env.env.contact_dim
    policy = config.configure_ddpg(dims=dims,
                                   params=params,
                                   clip_return=clip_return,
                                   process_type = process_type,
                                   feature_dim = feature_dim,
                                   fixed_num_of_contact = fixed_num_of_contact,
                                   contact_dim = contact_dim,
                                   pre_train_model = pre_train_model,
                                   include_action = include_action,
                                   update_model = update_model,
                                   feature_net_path = feature_net_path)
    if load_path is not None:
        # load_path = '/home/vioichigo/try/tactile-baselines/saved90.pkl'
        # load_path = './data/s3/test/save-std-noise/test/save-std-noise-1591730661664/saved160.pkl'
        load_path = '/home/vioichigo/try/tactile-baselines/saved_model/saved160.pkl'
        # tf_util.load_variables(load_path + '/saved.pkl')
        variables = [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = '') if 'process' not in x.name]
        tf_util.load_variables(load_path, variables = variables)
        # tf_util.load_variables(load_path, variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'ddpg/o_stats'))
        restored_step = policy.sess.run(policy.global_step)
        print("restored from step ", restored_step)

    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
    }

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]


    rollout_worker = RolloutWorker(env, policy, dims, logger, monitor=True, **rollout_params)
    evaluator = RolloutWorker(eval_env, policy, dims, logger, **eval_params)

    n_cycles = params['n_cycles']
    n_epochs = total_timesteps // n_cycles // rollout_worker.T // rollout_worker.rollout_batch_size
    print("n_epochs: ", n_epochs)


    return train(
        save_path=save_path, policy=policy, rollout_worker=rollout_worker,
        evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],
        n_cycles=params['n_cycles'], n_batches=params['n_batches'],
        policy_save_interval=policy_save_interval, demo_file=demo_file, exp_dir = exp_dir)


@click.command()
@click.option('--env', type=str, default='FetchReach-v1', help='the name of the OpenAI Gym environment that you want to train on')
@click.option('--total_timesteps', type=int, default=int(5e5), help='the number of timesteps to run')
@click.option('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code')
@click.option('--policy_save_interval', type=int, default=5, help='the interval with which policy pickles are saved. If set to 0, only the best and latest policy will be pickled.')
@click.option('--replay_strategy', type=click.Choice(['future', 'none']), default='future', help='the HER replay strategy to be used. "future" uses HER, "none" disables HER.')
@click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped')
@click.option('--demo_file', type=str, default = 'PATH/TO/DEMO/DATA/FILE.npz', help='demo data file path')
def main(**kwargs):
    learn(**kwargs)


if __name__ == '__main__':
    main()
