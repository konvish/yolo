#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by kong on 2020/9/2

import tensorflow as tf
import os


def find_trainable_variables(key):
    """
    获取tf的变量
    :param key: 变量名
    :return: tensor
    """
    with tf.variable_scope(key):
        return tf.trainable_variables()


def make_path(f):
    """
    创建目录
    :param f:文件名
    :return:
    """
    return os.makedirs(f, exist_ok=True)


def discount_with_dones(rewards, dones, gamma):
    """
    反馈折旧
    :param rewards: 反馈
    :param dones: 是否完成
    :param gamma: gamma
    :return:
    """
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r * (1. - done)
        discounted.append(r)
    return discounted[::-1]


def mse(pred, target):
    """
    平方误差
    :param pred: 预测
    :param target: 实际值
    :return: mse
    """
    return tf.square(pred - target) / 2.
