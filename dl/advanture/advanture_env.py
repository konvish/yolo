#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by kong on 2020/9/16

import cv2
import gym
import retro
import numpy as np
from baselines.common.atari_wrappers import FrameStack

from dl.sonic.sonic_env import PreprocessFrame, RewardScalar, AllowBacktracking

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
