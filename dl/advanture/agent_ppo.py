#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by kong on 2020/9/16

import retro
import tensorflow as tf
from baselines.common.vec_env import SubprocVecEnv
import numpy as np

from dl.utils import architecture as policies, PPOModel
from dl.advanture import advanture_env


def train():
    # AdventureIslandII-Nes,AdventureIsland3-Nes
    env = advanture_env.make_env(0)
    with tf.Session():
        PPOModel.learn(policy=policies.PPOPolicy, env=env, nsteps=2048,
                       total_timesteps=10000000,
                       gamma=0.99, lam=0.95, vf_coef=0.5, ent_coef=0.01, lr=2e-4, max_grad_norm=0.5, log_interval=10,
                       name='adventure_island3', cliprange=lambda _: 0.2, update=4)
        # PPOModel.learn(policy=policies.PPOPolicy, env=SubprocVecEnv([env]), nsteps=2048,
        #                total_timesteps=10000000,
        #                gamma=0.99, lam=0.95, vf_coef=0.5, ent_coef=0.01, lr=2e-4, max_grad_norm=0.5, log_interval=10,
        #                cliprange=lambda _: 0.1,
        #                name='adventure_island3')


if __name__ == '__main__':
    train()
