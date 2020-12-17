from envs import gym
import numpy as np
from pdb import set_trace as st
from envs.gym.envs.robotics.utils import process_data
import time

class EnvSpec(object):
    def __init__(self, obs_dim, act_dim, horizon, num_agents):
        self.observation_dim = obs_dim
        self.action_dim = act_dim
        self.horizon = horizon
        self.num_agents = num_agents

class GymEnv(object):
    def __init__(self, env_name):
        env = gym.make(env_name)
        self.env = env
        self.env_id = env.spec.id
        self.sim = env.env.sim

        self._horizon = env.spec.timestep_limit

        try:
            self._action_dim = self.env.env.action_dim
        except AttributeError:
            self._action_dim = self.env.env.action_space.shape[0]

        self._observation_dim = self.env.env.obs_dim

        try:
            self._num_agents = self.env.env.num_agents
        except AttributeError:
            self._num_agents = 1

        # Specs
        self.spec = EnvSpec(self._observation_dim, self._action_dim, self._horizon, self._num_agents)

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def observation_dim(self):
        return self._observation_dim

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def horizon(self):
        return self._horizon

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()

    def visualize_policy(self, policy, horizon=1000, num_episodes=1, mode='exploration'):
        try:
            self.env.env.visualize_policy(policy, horizon, num_episodes, mode)
        except:
            for ep in range(num_episodes):
                o = self.env.reset()
                d = False
                t = 0
                while t < horizon and d is False:
                    a = policy.get_action(o)[0] if mode == 'exploration' else policy.get_action(o)[1]['evaluation']
                    o, r, d, _ = self.env.step(a)
                    self.env.render()
                    t = t+1



    def evaluate_policy(self, policy,
                        num_episodes=5,
                        horizon=None,
                        gamma=1,
                        visual=False,
                        percentile=[],
                        get_full_dist=False,
                        mean_action=False,
                        init_state=None,
                        terminate_at_done=True,
                        save_video_location=None,
                        seed=None,
                        fixed_num_of_contact=0,
                        # object_index = 0,
                        obs_type = 'original'):

        if seed is not None:
            self.env._seed(seed)
        horizon = self._horizon if horizon is None else horizon
        mean_eval, std, min_eval, max_eval = 0.0, 0.0, -1e8, -1e8
        ep_returns = np.zeros(num_episodes)

        if save_video_location != None:
            self.env.monitor.start(save_video_location, force=True)

        for ep in range(num_episodes):

            if init_state is not None:
                o = self.reset()
                self.env.set_state(init_state[0], init_state[1])
                o = self.env._get_obs()
            else:
                o = self.reset()

            t, done = 0, False
            object_start = self.env.env.env.object_start
            while t < horizon and (done == False or terminate_at_done == False):
                if obs_type == 'joint':
                    processed = o[:object_start]
                elif obs_type == 'object':
                    processed = o[object_start:]
                elif obs_type == 'original':
                    processed = o
                # else:
                #     sim = self.sim
                #     num_of_contacts = sim.data.ncon
                #     total_num_of_geoms = sim.model.ngeom
                #     contacts = [sim.data.contact[i] for i in range(num_of_contacts)]
                #     geom1s = [contact.geom1 for contact in contacts]
                #     geom2s = [contact.geom2 for contact in contacts]
                #     geom1_names = [sim.model.geom_id2name(geom1) for geom1 in geom1s]
                #     geom2_names = [sim.model.geom_id2name(geom2) for geom2 in geom2s]
                #     geom2_bodys = [sim.model.geom_bodyid[geom2] for geom2 in geom2s]
                #     positions = [contact.pos for contact in contacts]
                #     forces = [sim.data.cfrc_ext[geom2_body] for geom2_body in geom2_bodys]
                #     processed = process_data(total_num_of_geoms,
                #                              num_of_contacts,
                #                              geom1s,
                #                              geom2s,
                #                              geom1_names,
                #                              geom2_names,
                #                              positions,
                #                              forces,
                #                              fixed_num_of_contact)
                #     processed = processed.reshape((-1))
                #     if obs_type == 'contact':
                #         processed = np.concatenate((o[object_start:], processed), axis = 0)

                if visual == True:
                    self.render()
                if mean_action:
                    a = policy.get_action(processed)[1]['mean']
                else:
                    a = policy.get_action(processed)[0]

                o, r, done, _ = self.step(a)
                ep_returns[ep] += (gamma ** t) * r
                t += 1

        if save_video_location != None:
            self.env.monitor.close()

        mean_eval, std = np.mean(ep_returns), np.std(ep_returns)
        min_eval, max_eval = np.amin(ep_returns), np.amax(ep_returns)
        base_stats = [mean_eval, std, min_eval, max_eval]

        percentile_stats = []
        full_dist = []

        for p in percentile:
            percentile_stats.append(np.percentile(ep_returns, p))

        if get_full_dist == True:
            full_dist = ep_returns

        return [base_stats, percentile_stats, full_dist]
