#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by kong on 2020/9/2

import tensorflow as tf
import os
from dl.sonic import sonic_env as env
from dl.utils import architecture as policies, PPOModel
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv


def main():
    config = tf.ConfigProto()
    os.environ["CUDA_VISBLE_DEVICES"] = "0"
    config.gpr_options.allow_growth = True
    with tf.Session(config=config):
        PPOModel.play(policy=policies.PPOPolicy, env=DummyVecEnv([env.make_train_1]), update=120)


if __name__ == '__main__':
    main()
