#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by kong on 2020/9/16

import cv2
import gym
import retro
import numpy as np
from baselines.common.atari_wrappers import FrameStack
import time

from dl.sonic.sonic_env import PreprocessFrame, RewardScalar

cv2.ocl.setUseOpenCL(False)


class ActionsDiscretizer(gym.ActionWrapper):
    def __init__(self, env):
        super(ActionsDiscretizer, self).__init__(env)
        buttons = ["B", None, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A"]
        actions = [['LEFT'], ['RIGHT'], ['UP'], ['DOWN'], ['A'], ['B'], ['A', 'B']]
        self._actions = []

        for action in actions:
            arr = np.array([False] * 9)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, action):
        return self._actions[action].copy()


class AllowBacktracking(gym.Wrapper):
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._last_x = 0
        self._last_time = time.time()

    def reset(self, **kwargs):
        self._cur_x = 0
        self._last_x = 0
        self._last_time = time.time()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        print(info)
        # self._cur_x = info['x']
        # # 右走奖励
        # x_reward = self._cur_x - self._last_x
        # # 速度奖励
        # cur_time = time.time()
        # s_reward = min(abs(float(x_reward) * 0.01 / (cur_time - self._last_time)), 10.0)
        # self._last_time = cur_time
        #
        # self._last_x = self._cur_x
        # if x_reward < -10 or x_reward > 5:
        #     x_reward = 0
        # # 死亡奖励
        # d_reward = -100 * (info['prev_lives'] - info['lives'])
        #
        # reward = x_reward + d_reward + rew + s_reward
        return obs, rew, done, info


def make_env(env_idx):
    dicts = [
        {'game': 'AdventureIsland3-Nes', 'state': 'Level1.state'},
        {'game': 'AdventureIslandII-Nes', 'state': 'FernIsland.Level1.state'}
    ]

    print(dicts[env_idx]['game'], dicts[env_idx]['state'], flush=True)
    env = retro.make(dicts[env_idx]['game'])
    env = ActionsDiscretizer(env)
    env = RewardScalar(env)
    env = PreprocessFrame(env)
    env = FrameStack(env, 4)
    env = AllowBacktracking(env)
    return env
