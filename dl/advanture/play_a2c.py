#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by kong on 2020/9/16

import tensorflow as tf
from dl.utils import A2CModel, architecture as policies
from dl.advanture import advanture_env


def main():
    # config = tf.ConfigProto()
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # config.gpu_options.allow_growth = True
    env = advanture_env.make_env(0)
    with tf.Session():  # config=config):
        A2CModel.play(policy=policies.A2CPolicy, env=env, update=6, name="adventure_island3")


if __name__ == '__main__':
    main()
