#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by kong on 2020/9/2

import tensorflow as tf
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from dl.sonic import sonic_env as env
from dl.utils import architecture as policies, PPOModel


def main():
    # config = tf.ConfigProto()
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # config.gpu_options.allow_growth = True
    with tf.Session():  # config=config):
        subEnv = SubprocVecEnv([
            # env.make_train_0,
            # env.make_train_1,
            # env.make_train_2,
            env.make_train_3
            # env.make_train_4,
            # env.make_train_5,
            # env.make_train_6,
            # env.make_train_7,
            # env.make_train_8,
            # env.make_train_9,
            # env.make_train_10,
            # env.make_train_11,
            # env.make_train_12
        ])
        dummyEnv = DummyVecEnv([env.make_train_3])
        PPOModel.learn(policy=policies.PPOPolicy, env=subEnv,
                       nsteps=2048,
                       total_timesteps=10000000,
                       gamma=0.99,
                       lam=0.95,
                       vf_coef=0.5,
                       ent_coef=0.01,
                       lr=lambda _: 2e-4,
                       cliprange=lambda _: 0.2,
                       max_grad_norm=0.5,
                       nenvs=subEnv.num_envs,
                       log_interval=10, update=20, test_env=dummyEnv)


if __name__ == '__main__':
    main()
