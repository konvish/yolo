#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by kong on 2020/9/4

import tensorflow as tf
import retro
from baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from dl.utils import architecture as policies, A2CModel, PPOModel


def train():
    # AdventureIslandII-Nes,AdventureIsland3-Nes
    env = retro.make("AdventureIsland3-Nes")
    with tf.Session():
        A2CModel.learn(policy=policies.A2CPolicy, env=SubprocVecEnv([env]), nsteps=2048,
                       total_timesteps=10000000,
                       gamma=0.99, lam=0.95, vf_coef=0.5, ent_coef=0.01, lr=2e-4, max_grad_norm=0.5, log_interval=10,
                       name='adventure_island3')
        PPOModel.learn(policy=policies.PPOPolicy, env=SubprocVecEnv([env]), nsteps=2048,
                       total_timesteps=10000000,
                       gamma=0.99, lam=0.95, vf_coef=0.5, ent_coef=0.01, lr=2e-4, max_grad_norm=0.5, log_interval=10,
                       cliprange=lambda _: 0.1,
                       name='adventure_island3')


def test():
    env = retro.make("AdventureIsland3-Nes")
    PPOModel.play(policy=policies.PPOPolicy, env=DummyVecEnv([env]), update=20, name='adventure_island3')
    A2CModel.play(policy=policies.A2CPolicy, env=DummyVecEnv([env]), update=20, name='adventure_island3')


if __name__ == '__main__':
    train()
