import numpy as np
import scipy as sp
import scipy.sparse.linalg as spLA
import copy
import time as timer
import copy
from tactile_baselines import logger
import tensorflow as tf
from pdb import set_trace as st


class NPG(Base):
    def __init__(self,
                 env,
                 policy,
                 baseline,
                 normalized_step_size=0.01,
                 const_learn_rate=None,
                 FIM_invert_args={'iters': 10, 'damping': 1e-4},
                 hvp_sample_frac=1.0,
                 seed=123,
                 save_logs=False,
                 kl_dist=None,
                 input_normalization=None,
                 **kwargs
                 ):
        """
        All inputs are expected in mjrl's format unless specified
        :param normalized_step_size: Normalized step size (under the KL metric). Twice the desired KL distance
        :param kl_dist: desired KL distance between steps. Overrides normalized_step_size.
        :param const_learn_rate: A constant learn rate under the L2 metric (won't work very well)
        :param FIM_invert_args: {'iters': # cg iters, 'damping': regularization amount when solving with CG
        :param hvp_sample_frac: fraction of samples (>0 and <=1) to use for the Fisher metric (start with 1 and reduce if code too slow)
        :param seed: random seed
        """

        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.alpha = const_learn_rate
        self.n_step_size = normalized_step_size if kl_dist is None else 2.0 * kl_dist
        self.seed = seed
        self.FIM_invert_args = FIM_invert_args
        self.hvp_subsample = hvp_sample_frac
        self.running_score = None

        self.build_graph()


    def build_graph(self):
        self.obs_ph, self.action_ph, self.next_obs_ph, self.adv_ph,
                            self.dist_info_ph_dict, self.all_phs_dict =
                            self._make_input_placeholders('', recurrent=False, next_obs=True)

        self.surr = self.CPI_surrogate()
        self.vpg_grad = self.flat_vpg()
        self.hvp = self.HVP()

    def HVP(self):

        self.vec = tf.placeholder(tf.float32, shape=[None], name='vec')
        self.regu_coef = tf.placeholder(tf.float32, shape=[1], name='regu_coef')
        observations = self.obs_ph
        actions = self.action_ph
        if self.hvp_subsample is not None and self.hvp_subsample < 0.99:
            num_samples = observations.shape[0]
            idx = int(self.hvp_subsample*num_samples)
            obs = tf.random.shuffle(observations)[:idx]
            act = tf.random.shuffle(actions)[:idx]
        else:
            obs = observations
            act = actions

        old_dist_info = self.policy.old_dist_info(obs, act)
        new_dist_info = self.policy.new_dist_info(obs, act)
        mean_kl = self.policy.mean_kl(new_dist_info, old_dist_info)
        grad_fo = tf.gradients(mean_kl, self.policy.trainable_params)
        flat_grad = tf.reshape(grad_fo, [-1])
        h = tf.sum(flat_grad*self.vec)
        hvp = tf.gradients(h, self.policy.trainable_params)
        hvp_flat = tf.reshape(hvp, [-1])
        return hvp_flat + regu_coef * self.vec

    def build_Hvp_eval(self, observations, actions, regu_coef=None):
        def eval(v):
            st()
            print(vec.shape)
            regu_coef = self.FIM_invert_args['damping'] if regu_coef is None else regu_coef
            self.npg_grad = self.cg_solve(self.hvp, self.vpg_grad, x_0=self.vpg_grad.copy(),
                                cg_iters=self.FIM_invert_args['iters'])
            Hvp, npg_grad = self.sess.run([self.hvp, self.npg_grad], {self.obs_ph: observations,
                                             self.action_ph: actions,
                                             self.vec: vec,
                                             self.regu_coef: regu_coef})

            return Hvp, npg_grad

        return eval

    def CPI_surrogate(self):
        old_dist_info = self.policy.old_dist_info(self.obs_ph, self.action_ph)
        new_dist_info = self.policy.new_dist_info(self.obs_ph, self.action_ph)
        LR = self.policy.likelihood_ratio(new_dist_info, old_dist_info)
        surr = torch.mean(LR*self.adv_ph)
        return surr

    def flat_vpg(self):
        cpi_surr = self.surr
        vpg_grad = tf.gradients(cpi_surr, var_list = self.policy.trainable_params)
        vpg_grad = tf.reshape(vpg_grad, [-1])
        return vpg_grad

    def cg_solve(self, f_Ax, b, x_0=None, cg_iters=10, residual_tol=1e-10):
        x = tf.zeros(tf.shape(b))
        r = b.copy()
        p = r.copy()
        rdotr = tf.tensordot(r, r, 1)

        for i in range(cg_iters):
            z = f_Ax(p)
            v = rdotr / tf.tensordot(p, z, 1)
            x += tf.matmul(v,p)
            r -= tf.matmul(v,z)
            newrdotr = tf.tensordot(r, r, 1)
            mu = newrdotr / rdotr
            p = r + mu * p

            rdotr = newrdotr
            if rdotr < residual_tol:
                break
        return x


    def optimize_policy(self, samples):
        t_gLL = 0.0
        t_FIM = 0.0

        # Optimization algorithm
        # --------------------------
        observations = samples['observations']
        actions = samples['actions']
        advantages = samples['advantages']

        surr_before = self.sess.run([self.surr], feed_dict = {self.obs_ph: observations,
                                                              self.action_ph: actions,
                                                              self.adv_ph: advantages})

        # VPG
        vpg_grad = self.sess.run([self.vpg_grad], feed_dict = {self.obs_ph: observations,
                                                              self.action_ph: actions,
                                                              self.adv_ph: advantages})
        # NPG
        hvp, npg_grad = self.build_Hvp_eval([observations, actions], regu_coef=self.FIM_invert_args['damping'])


        # Step size computation
        # --------------------------
        if self.alpha is not None:
            alpha = self.alpha
            n_step_size = (alpha ** 2) * np.dot(vpg_grad.T, npg_grad)
        else:
            n_step_size = self.n_step_size
            alpha = np.sqrt(np.abs(self.n_step_size / (np.dot(vpg_grad.T, npg_grad) + 1e-20)))

        curr_params = self.policy.get_param_values()
        new_params = curr_params + alpha * npg_grad
        self.policy.set_param_values(new_params, set_new=True, set_old=False)
        surr_after = self.sess.run([self.surr], feed_dict = {self.obs_ph: observations,
                                                              self.action_ph: actions,
                                                              self.adv_ph: advantages})

        self.policy.set_param_values(new_params, set_new=True, set_old=True)
