#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by kong on 2020/9/2

import tensorflow as tf
import os
from dl.utils import A2CModel
import dl.utils.architecture as policies
import dl.sonic.sonic_env as env
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv


def main():
    # config = tf.ConfigProto()
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # config.gpu_options.allow_growth = True
    with tf.Session():  # config=config):
        A2CModel.play(policy=policies.A2CPolicy, env=DummyVecEnv([env.make_train_2]), update=11)


if __name__ == '__main__':
    main()
