#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by kong on 2020/9/16

import tensorflow as tf
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

from dl.advanture import advanture_env
from dl.utils import architecture as policies, PPOModel


def main():
    # config = tf.ConfigProto()
    # os.environ["CUDA_VISBLE_DEVICES"] = "0"
    # config.gpr_options.allow_growth = True
    env = advanture_env.make_env(0)
    with tf.Session():  # config=config):
        PPOModel.play(policy=policies.PPOPolicy, env=env, update=12)


if __name__ == '__main__':
    main()
