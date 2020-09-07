#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by kong on 2020/8/28
import tensorflow as tf
from collections import deque
import numpy as np


class PGNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='PGNet'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        with tf.variable_scope(name):
            with tf.name_scope("inputs"):
                self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name='inputs_')
                self.actions = tf.placeholder(tf.int32, [None, action_size], name="actions")
                self.discounted_episode_rewards_ = tf.placeholder(tf.float32, [None, ],
                                                                  name="discounted_episode_rewards_")
                self.mean_reward_ = tf.placeholder(tf.float32, name='mean_reward')

            with tf.name_scope("conv1"):
                self.conv1 = tf.layers.conv2d(inputs=self.inputs_,
                                              filters=32,
                                              kernel_size=[8, 8],
                                              strides=[4, 4],
                                              padding='VALID',
                                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                              name='conv1')
                self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1, training=True, epsilon=1e-5,
                                                                     name='batch_norm1')
                self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name='conv1_out')
                # --> [20,20,32]
            with tf.name_scope("conv2"):
                self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                              filters=64,
                                              kernel_size=[4, 4],
                                              strides=[2, 2],
                                              padding='valid',
                                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                              name='conv2')
                self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2, training=True, epsilon=1e-5,
                                                                     name='batch_norm2')
                self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name='conv2_out')
                # --> [9,9,64]

            with tf.name_scope("conv3"):
                self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                                              filters=128,
                                              kernel_size=[4, 4],
                                              strides=[2, 2],
                                              padding='valid',
                                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                              name='conv3')
                self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3, training=True, epsilon=1e-5,
                                                                     name='batch_norm3')
                self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name='conv3_out')
                # --> [3,3,128]

            with tf.name_scope("flatten"):
                self.flatten = tf.layers.flatten(self.conv3_out)
                # --> [1152]

            with tf.name_scope("fc1"):
                self.fc = tf.layers.dense(inputs=self.flatten, units=512, activation=tf.nn.elu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(), name='fc1')
            with tf.name_scope("logits"):
                self.logits = tf.layers.dense(inputs=self.fc, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              units=3, activation=None)

            with tf.name_scope("sofxmax"):
                self.action_distribution = tf.nn.softmax(self.logits)
            with tf.name_scope("loss"):
                self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.actions)
                self.loss = tf.reduce_mean(self.neg_log_prob * self.discounted_episode_rewards_)
            with tf.name_scope("train"):
                self.train_opt = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)


class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNet'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name='inputs_')
            self.actions_ = tf.placeholder(tf.float32, [None, 3], name="actions_")
            self.target_Q = tf.placeholder(tf.float32, [None], name='target')

            self.conv1 = tf.layers.conv2d(inputs=self.inputs_,
                                          filters=32,
                                          kernel_size=[8, 8],
                                          strides=[4, 4],
                                          padding='VALID',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name='conv1')
            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1, training=True, epsilon=1e-5,
                                                                 name='batch_norm1')
            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name='conv1_out')
            # --> [20,20,32]
            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                          filters=64,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding='valid',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name='conv2')
            self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2, training=True, epsilon=1e-5,
                                                                 name='batch_norm2')
            self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name='conv2_out')
            # --> [9,9,64]

            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                                          filters=128,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding='valid',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name='conv3')
            self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3, training=True, epsilon=1e-5,
                                                                 name='batch_norm3')
            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name='conv3_out')
            # --> [3,3,128]

            self.flatten = tf.layers.flatten(self.conv3_out)
            # --> [1152]

            self.fc = tf.layers.dense(inputs=self.flatten, units=512, activation=tf.nn.elu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(), name='fc1')
            self.output = tf.layers.dense(inputs=self.fc, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units=3, activation=None)

            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)


class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size), size=batch_size, replace=False)
        return [self.buffer[i] for i in index]


class DDDQNNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DDDQNNet'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = name

        with tf.name_scope(self.name):
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.ISWeights_ = tf.placeholder(tf.float32, [None, 1], name="IS_weights")
            self.actions_ = tf.placeholder(tf.float32, [None, action_size], name="actions_")
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            self.conv1 = tf.layers.conv2d(inputs=self.inputs_, filters=32, kernel_size=[8, 8], strides=[4, 4],
                                          padding='valid',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name='conv1')
            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")

            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out, filters=64, kernel_size=[4, 4], strides=[2, 2],
                                          padding='valid',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name='conv2')
            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")

            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out, filters=32, kernel_size=[8, 8], strides=[4, 4],
                                          padding='valid',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name='conv3')
            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")

            self.flatten = tf.layers.flatten(self.conv3_out)

            self.value_fc = tf.layers.dense(inputs=self.flatten, units=512, activation=tf.nn.elu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='value_fc')

            self.value = tf.layers.dense(inputs=self.value_fc, units=1, activation=None,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(), name="value")
            self.advantage_fc = tf.layers.dense(inputs=self.flatten, units=512, activation=tf.nn.elu,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                name='advantage_fc')
            self.advantage = tf.layers.dense(inputs=self.advantage_fc, units=self.action_size, activation=None,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             name='advantages')
            self.output = self.value + tf.subtract(self.advantage,
                                                   tf.reduce_mean(self.advantage, axis=1, keepdims=True))
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)
            self.absolute_errors = tf.abs(self.target_Q - self.Q)
            self.loss = tf.reduce_mean(self.ISWeights_ * tf.squared_difference(self.target_Q, self.Q))
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)


class SumTree(object):
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority, data):
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        parent_index = 0
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    def total_priority(self):
        return self.tree[0]


class Memory2(object):
    PER_e = 0.01
    PER_a = 0.6
    PER_b = 0.4
    PER_b_increment_per_sampling = 0.001
    absolute_error_upper = 1.

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, experience):
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        self.tree.add(max_priority, experience)

    def sample(self, n):
        memory_b = []
        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)
        priority_segment = self.tree.total_priority() / n
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority()
        max_weight = (p_min * n) ** (-self.PER_b)

        for i in range(n):
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            index, priority, data = self.tree.get_leaf(value)
            sampling_probabilities = priority / self.tree.total_priority()
            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b) / max_weight
            b_idx[i] = index
            experience = [data]
            memory_b.append(experience)

        return b_idx, memory_b, b_ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
