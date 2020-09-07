#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by kong on 2020/9/1

import numpy as np
import tensorflow as tf
# actions的概率分布
from baselines.common.distributions import make_pdtype


def conv_layer(inputs, filters, kernel_size, strides, gain=1.0):
    """
    卷积网络
    :param inputs: 输入
    :param filters: channel
    :param kernel_size: 卷积核大小
    :param strides: 滑步
    :param gain: 卷积初始化模式
    :return: 卷积层
    """
    return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=(strides, strides),
                            activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer(gain=gain))


def fc_layer(inputs, units, activation_fn=tf.nn.relu, gain=1.0):
    """
    全连接网络
    :param inputs: 输入
    :param units: 输出大小
    :param activation_fn: 激活函数
    :param gain: 卷积初始化模式
    :return: 全连接
    """
    return tf.layers.dense(inputs=inputs, units=units, activation=activation_fn,
                           kernel_initializer=tf.orthogonal_initializer(gain))


class A2CPolicy(object):
    def __init__(self, sess, ob_space, action_space, nbatch, nsteps, reuse=False):
        """
        创建A2C模型
                       - 1层全连接（policy）
        3层CNN+1层全连接-
                       - 1层全连接（value）
        :param sess: tf.session
        :param ob_space: 状态
        :param action_space: 动作
        :param nbatch: batch
        :param nsteps: step
        :param reuse: reuse
        """
        gain = np.sqrt(2)
        self.pdtype = make_pdtype(action_space)
        height, weight, channel = ob_space.shape
        ob_shape = (height, weight, channel)
        inputs_ = tf.placeholder(tf.float32, [None, *ob_shape], name="input")

        scaled_images = tf.cast(inputs_, tf.float32) / 255.

        with tf.variable_scope("model", reuse=reuse):
            conv1 = conv_layer(scaled_images, 32, 8, 4, gain)
            conv2 = conv_layer(conv1, 64, 4, 2, gain)
            conv3 = conv_layer(conv2, 64, 3, 1, gain)
            flatten1 = tf.layers.flatten(conv3)
            fc_common = fc_layer(flatten1, 512, gain=gain)

            # actions prob
            self.pd, self.pi = self.pdtype.pdfromlatent(fc_common, init_scale=0.01)
            # value
            vf = fc_layer(fc_common, 1, activation_fn=None)[:, 0]

        self.initial_state = None
        # 按概率分布挑选动作
        a0 = self.pd.sample()

        def step(state_in, *_args, **_kwargs):
            """
            获取状态对应的动作与价值
            :param state_in: 状态
            :param _args: 参数
            :param _kwargs: 参数
            :return: a,v
            """
            action, value = sess.run([a0, vf], {inputs_: state_in})
            return action, value

        def value(state_in, *_args, **_kwargs):
            """
            获取状态对应的价值
            :param state_in: 状态
            :param _args: 参数
            :param _kwargs: 参数
            :return: v
            """
            return sess.run(vf, {inputs_: state_in})

        def select_action(state_in, *_args, **_kwargs):
            """
            获取状态的对应动作
            :param state_in: 状态
            :param _args: 参数
            :param _kwargs: 参数
            :return: a
            """
            return sess.run(a0, {inputs_: state_in})

        self.inputs_ = inputs_
        self.vf = vf
        self.step = step
        self.value = value
        self.select_action = select_action


class PPOPolicy(object):
    def __init__(self, sess, ob_space, action_space, nbatch, nsteps, reuse=False):
        """
        创建PPO模型=A2C+近端策略优化
                        - 1层全连接（policy）
        3层CNN+1层全连接-
                        - 1层全连接（value）
        :param sess: tf.session
        :param ob_space: 状态
        :param action_space: 动作
        :param nbatch: batch
        :param nsteps: step
        :param reuse: reuse
        """
        gain = np.sqrt(2)
        self.pdtype = make_pdtype(action_space)
        height, weight, channel = ob_space.shape
        ob_shape = (height, weight, channel)

        inputs_ = tf.placeholder(tf.float32, [None, *ob_shape], name="input")
        scaled_images = tf.cast(inputs_, tf.float32) / 255.

        with tf.variable_scope("model", reuse=reuse):
            conv1 = conv_layer(scaled_images, 32, 8, 4, gain)
            conv2 = conv_layer(conv1, 64, 4, 2, gain)
            conv3 = conv_layer(conv2, 64, 3, 1, gain)
            flatten1 = tf.layers.flatten(conv3)
            fc_common = fc_layer(flatten1, 512, gain=gain)

            self.pd, self.pi = self.pdtype.pdfromlatent(fc_common, init_scale=0.01)
            vf = fc_layer(fc_common, 1, activation_fn=None)[:, 0]
        self.initial_state = None
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)

        def step(state_in, *_args, **_kwargs):
            """
            获取状态对应的动作与价值
            :param state_in: 状态
            :param _args: 参数
            :param _kwargs: 参数
            :return: a,v,neg log p
            """
            return sess.run([a0, vf, neglogp0], {inputs_: state_in})

        def value(state_in, *_args, **_kwargs):
            """
            获取状态对应的价值
            :param state_in: 状态
            :param _args: 参数
            :param _kwargs: 参数
            :return: v
            """
            return sess.run(vf, {inputs_: state_in})

        def select_action(state_in, *_args, **_kwargs):
            """
            获取状态的对应动作
            :param state_in: 状态
            :param _args: 参数
            :param _kwargs: 参数
            :return: a
            """
            return sess.run(a0, {inputs_: state_in})

        self.inputs_ = inputs_
        self.vf = vf
        self.step = step
        self.value = value
        self.select_action = select_action
