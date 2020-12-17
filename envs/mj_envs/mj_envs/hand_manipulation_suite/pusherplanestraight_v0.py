import numpy as np
from envs.gym import utils
from envs.mj_envs.mj_envs import mujoco_env
import os
from copy import copy
from envs.gym import spaces
import cv2
import time
from pdb import set_trace as st
# from mjrl.utils.utils import *
from mujoco_py import MjViewer



class PusherPlaneStraightEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, max_episode_steps=None):
        self.object_radius = 0.03
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0
        high = np.array([0.1, 0.1])
        self.low = -high[0]
        self.high = high[0]
        self.action_space = spaces.Box(low=-high, high=high)
        utils.EzPickle.__init__(self)
        fullpath = os.path.join(os.path.dirname(__file__), "assets", 'pusher_plane_straight.xml')
        # mujoco_e.MujocoEnv.__init__(self, fullpath, 2)
        mujoco_env.MujocoEnv.__init__(self, fullpath, 2)
        self.action_space = spaces.Box(low=-high, high=high)
        self.cam_name = 'top_cam'
        self.frame_skip = 5
        self.object_start = 8

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = 0
        self.sim.forward()
        self.viewer.cam.distance = 1.5

    def _step(self, a):
        u_clipped = np.clip(a, self.low, self.high)[:2]

        vec_1 = self.get_body_com("object") - self.get_body_com("pusher")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")

        reward_near = - np.linalg.norm(vec_1[:2])
        reward_dist = - np.linalg.norm(vec_2[:2])
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + 0.001 * reward_ctrl + 0.1 * reward_near

        qpos = self.data.qpos
        qvel = self.data.qvel
        qvel[:2] = np.array(u_clipped)[:]

        self.do_simulation(np.zeros(self.model.nu), self.frame_skip)

        # Approximating quasi-static pushing
        if np.linalg.norm(vec_1[:2])>self.object_radius:
            qpos = self.data.qpos
            qvel = self.data.qvel
            qvel[2:5] = np.zeros(3)
            self.set_state(qpos, qvel)
        ob = self._get_obs()

        self._elapsed_steps += 1
        if self._past_limit():
            if self.metadata.get('semantics.autoreset'): # This should usually be false
                raise NotImplementedError('CHECK WHY IS AUTORESET SET TO TRUE')
                _ = self.reset() # automatically reset the env
            done = True
        else:
            done = False

        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = 0
        self.sim.forward()
        self.viewer.cam.distance = 1.5


    def reset_model(self):
        self._elapsed_steps = 0
        pusher_position = np.array([-0.30, 0.])
        init_block_position = np.array([-0.15, 0., 0.015]) + np.random.uniform(-0.1, 0.1, 3)
        init_block_position[2] = 0.015
        goal_block_position = np.array([0., 0.])
        qpos = np.zeros(self.model.nq)
        qpos[:2] = pusher_position
        # object information
        qpos[2:5] = init_block_position
        qpos[5:7] = goal_block_position
        qvel = np.zeros(self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qvel.flat[:5],
            self.sim.data.qpos.flat[:5],
        ])

    def _past_limit(self):
        """Return true if we are past our limit"""
        if self._max_episode_steps is not None and self._max_episode_steps <= self._elapsed_steps:
            return True
        else:
            return False


class PusherStraight(PusherPlaneStraightEnv):

    def __init__(self, T=None, **kwargs):
        PusherPlaneStraightEnv.__init__(self, max_episode_steps=T)
        self.init_state = self.reset()
        self.Q = None
        self.R = None
        self.n_act = self.action_space.shape[0]
        self.n_obs = self.observation_space.shape[0]
        self.init_video_writing(**kwargs)

    def forward_obs_cost(self, x, u, *args, render=False, **kwargs):
        qpos = self.data.qpos
        qvel = self.data.qvel
        qpos[:5] = x[:5]
        qvel[:5] = x[5:]
        self.set_state(qpos, qvel)
        x_new, reward, _, _ = self.step(u)
        self.viz(render)
        return x_new, -reward

    def forward(self, *args, **kwargs):
        next_state, _ = self.forward_obs_cost(*args, **kwargs)
        return next_state

    def cost(self, *args, **kwargs):
        _, cost = self.forward_obs_cost(*args, **kwargs)
        return cost

    def init_video_writing(self, res=(1024,1024), fname=None):
        if fname==None:
            self.video=False
        else:
            self.video=True
            self.res = res
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(fname, fourcc, 5, self.res)

    def viz(self, render):
        if render==True:
            if self.video==False:
                self.render()
                time.sleep(0.02)
            if self.video==True:
                I = self.sim.render(*self.res, camera_name = self.cam_name)
                self.out.write(cv2.flip(I, 0))

    def save_video(self):
        self.out.release()

    def close(self):
        self.save_video()
