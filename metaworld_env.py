import collections
import random

import dm_env
import numpy as np
from dm_env._environment import TimeStep
import metaworld
from dm_env.specs import Array, BoundedArray
import cv2
import metaworld.policies

import dmc


class Env(dm_env.Environment):
    def __init__(self, env_name, im_width=84, im_height=84):
        self._observation = None
        env_set = metaworld.MT1(env_name)
        self._metaworld_env = env_set.train_classes[env_name]()
        self._tasks = [task for task in env_set.train_tasks if task.env_name == env_name]
        self.im_w = im_width
        self.im_h = im_height

        self.step_id = None
        self._pixels_key = 'pixels'

        self._metaworld_obs = None

    def reset(self) -> TimeStep:
        self._metaworld_env.set_task(random.choice(self._tasks))
        self._metaworld_obs = self._metaworld_env.reset()
        self.step_id = 0

        self._observation = self._metaworld_env.render(offscreen=True, resolution=(self.im_w, self.im_h))

        observation = collections.OrderedDict()
        observation[self._pixels_key] = np.array(self._observation, dtype=np.uint8)

        return dm_env.TimeStep(dm_env.StepType.FIRST, 0., 1., observation)

    def step(self, action) -> TimeStep:
        obs, reward, done, info = self._metaworld_env.step(action)
        self._metaworld_obs = obs
        self.step_id += 1

        self._observation = self._metaworld_env.render(offscreen=True, resolution=(self.im_w, self.im_h))

        observation = collections.OrderedDict()
        observation[self._pixels_key] = np.array(self._observation, dtype=np.uint8)

        step_type = dm_env.StepType.LAST if self.step_id >= self._metaworld_env.max_path_length else dm_env.StepType.MID

        return dm_env.TimeStep(step_type, reward, 1., observation)

    def observation_spec(self):
        observation_spec = collections.OrderedDict()
        observation_spec[self._pixels_key] = Array(shape=(self.im_h, self.im_w, 3), dtype=np.uint8,
                                                   name='observation')
        return observation_spec

    def action_spec(self):
        return BoundedArray(shape=self._metaworld_env.action_space.shape, dtype=self._metaworld_env.action_space.dtype, minimum=self._metaworld_env.action_space.low, maximum=self._metaworld_env.action_space.high)

    def __getattr__(self, name):
        return getattr(self._metaworld_env, name)

    def render(self, width=640, height=640, camera_id=None):
        return self._metaworld_env.render(offscreen=True, resolution=(width, height))

    @property
    def physics(self):
        return self

    def metaworld_obs(self):
        return self._metaworld_obs


class Expert:
    def __init__(self, policy, env):
        self.policy = policy
        self.env = env
        self.scale = 2 / (self.env._metaworld_env.action_space.high - self.env._metaworld_env.action_space.low)

    def train(self, *args, **kwargs):
        pass

    def act(self, *args, **kwargs):
        action = self.policy.get_action(self.env.metaworld_obs())
        action = self.scale * (action - self.env._metaworld_env.action_space.low) - 1
        return action


if __name__ == '__main__':
    env = Env('drawer-close-v2')
    env = dmc.wrap(env, frame_stack=3, action_repeat=2, episode_len=60)
    expert = Expert(metaworld.policies.SawyerDrawerCloseV2Policy(), env)
    timestep = env.reset()
    for i in range(120):
        cv2.imshow('Image', env.render())
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        timestep = env.step(expert.act())

