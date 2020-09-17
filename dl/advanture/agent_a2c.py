#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by kong on 2020/9/4

import retro
import tensorflow as tf
from baselines.common.vec_env import SubprocVecEnv
import numpy as np

from dl.utils import architecture as policies, A2CModel
from dl.advanture import advanture_env


def train():
    # AdventureIslandII-Nes,AdventureIsland3-Nes
    env = advanture_env.make_env(0)
    with tf.Session():
        A2CModel.learn(policy=policies.A2CPolicy, env=env, nsteps=3096,
                       total_timesteps=10000000,
                       gamma=0.99, lam=0.95, vf_coef=0.5, ent_coef=0.01, lr=2e-4, max_grad_norm=0.5, log_interval=10,
                       name='adventure_island3', update=12)
        # PPOModel.learn(policy=policies.PPOPolicy, env=SubprocVecEnv([env]), nsteps=2048,
        #                total_timesteps=10000000,
        #                gamma=0.99, lam=0.95, vf_coef=0.5, ent_coef=0.01, lr=2e-4, max_grad_norm=0.5, log_interval=10,
        #                cliprange=lambda _: 0.1,
        #                name='adventure_island3')


# def test():
#     env = retro.make("AdventureIsland3-Nes")
#     PPOModel.play(policy=policies.PPOPolicy, env=DummyVecEnv([env]), update=20, name='adventure_island3')
#     A2CModel.play(policy=policies.A2CPolicy, env=DummyVecEnv([env]), update=20, name='adventure_island3')


if __name__ == '__main__':
    train()
