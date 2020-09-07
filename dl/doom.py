#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by kong on 2020/8/27

import tensorflow as tf
import numpy as np
from vizdoom import *
import random
import time
from skimage import transform
from collections import deque
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
from dl.PGNetwork import PGNetwork


def create_environment():
    game = DoomGame()
    game.load_config('health_gathering.cfg')
    game.set_doom_scenario_path('health_gathering.wad')
    game.init()
    # actions:[[1,0,0],[0,1,0],[0,0,1]]
    possible_actions = np.identity(3, dtype=int).tolist()
    return game, possible_actions


def preprocess_frame(frame):
    cropped_frame = frame[80:, :]  # 切除屋顶部分
    normalized_frame = cropped_frame / 255.0  # 归一化
    preprocessed_frame = transform.resize(normalized_frame, [84, 84])  # 缩放为84*84
    return preprocessed_frame


stack_size = 4
# 初始化队列，大小为stack_size的84*84的图片
stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)


def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)
    if is_new_episode:
        # 清空数据
        stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)
    else:
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)
    return stacked_state, stacked_frames


game, possible_actions = create_environment()
state_size = [84, 84, 4]
action_size = game.get_available_buttons_size()
stack_size = 4
learning_rate = 0.002
num_epochs = 500
batch_size = 1000
gamma = 0.95
training = True


def discount_and_normalize_rewards(episode_rewards):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative
    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - mean) / std
    return discounted_episode_rewards


tf.reset_default_graph()
PGNet = PGNetwork(state_size, action_size, learning_rate)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

writer = tf.summary.FileWriter("/tensorboard/pg/test")
tf.summary.scalar("Loss", PGNet.loss)
tf.summary.scalar("Reward_mean", PGNet.mean_reward_)
write_op = tf.summary.merge_all()


def make_batch(batch_size, stacked_frames):
    states, actions, rewards_of_episode, rewards_of_batch, discounted_rewards = [], [], [], [], []
    episode_num = 1
    game.new_episode()
    state = game.get_state().screen_buffer
    state, stacked_frames = stack_frames(stacked_frames, state, True)
    while True:
        action_probability_distribution = sess.run(PGNet.action_distribution,
                                                   feed_dict={PGNet.inputs_: state.reshape(1, *state_size)})
        action = np.random.choice(range(action_probability_distribution.shape[1]),
                                  p=action_probability_distribution.ravel())
        action = possible_actions[action]
        reward = game.make_action(action)
        done = game.is_episode_finished()
        states.append(state)
        actions.append(action)
        rewards_of_episode.append(reward)
        if done:
            next_state = np.zeros((84, 84), dtype=np.int)
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            rewards_of_batch.append(rewards_of_episode)
            discounted_rewards.append(discount_and_normalize_rewards(rewards_of_episode))
            if len(np.concatenate(rewards_of_batch)) > batch_size:
                break
            rewards_of_episode = []
            episode_num += 1
            game.new_episode()
            state = game.get_state().screen_buffer
            state, stacked_frames = stack_frames(stacked_frames, state, True)
        else:
            next_state = game.get_state().screen_buffer
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            state = next_state
    return np.stack(np.array(states)), np.stack(np.array(actions)), np.concatenate(rewards_of_batch), np.concatenate(
        discounted_rewards), episode_num


allRewards = []
total_rewards = 0
maximumRewardRecorded = 0
mean_reward_total = []
epoch = 1
average_reward = []
saver = tf.train.Saver()
if training:
    # saver.restore(sess, "./models/doom.ckpt")
    while epoch < num_epochs + 1:
        states_mb, actions_mb, rewards_of_batch, discounted_rewards_mb, nb_episodes_mb = make_batch(batch_size,
                                                                                                    stacked_frames)
        total_reward_of_that_batch = np.sum(rewards_of_batch)
        allRewards.append(total_reward_of_that_batch)
        mean_reward_of_that_batch = np.divide(total_reward_of_that_batch, nb_episodes_mb)
        mean_reward_total.append(mean_reward_of_that_batch)
        average_reward_of_all_training = np.divide(np.sum(mean_reward_total), epoch)
        maximumRewardRecorded = np.amax(allRewards)
        print("==================================")
        print("Epoch: ", epoch, "/", num_epochs)
        print("----------------")
        print("Number of training episodes: {}".format(nb_episodes_mb))
        print("Total reward: {}".format(total_reward_of_that_batch, nb_episodes_mb))
        print("Mean Reward of that batch {}".format(mean_reward_of_that_batch))
        print("Average Reward of all training: {}".format(average_reward_of_all_training))
        print("Max reward for a batch so far: {}".format(maximumRewardRecorded))
        loss_, _ = sess.run([PGNet.loss, PGNet.train_opt], feed_dict={
            PGNet.inputs_: states_mb.reshape((len(states_mb), 84, 84, 4)),
            PGNet.actions: actions_mb,
            PGNet.discounted_episode_rewards_: discounted_rewards_mb
        })
        print("Training Loss: {}".format(loss_))
        summary = sess.run(write_op, feed_dict={
            PGNet.inputs_: states_mb.reshape((len(states_mb), 84, 84, 4)),
            PGNet.actions: actions_mb,
            PGNet.discounted_episode_rewards_: discounted_rewards_mb,
            PGNet.mean_reward_: mean_reward_of_that_batch
        })

        writer.add_summary(summary, epoch)
        writer.flush()

        if (epoch % 10) == 0:
            saver.save(sess, "./models/doom.ckpt")
            print("Model saved")
        epoch += 1

saver = tf.train.Saver()

with tf.Session() as sess:
    game = DoomGame()
    game.load_config("health_gathering.cfg")
    game.set_doom_scenario_path("health_gathering.wad")
    saver.restore(sess, "./models/doom.ckpt")
    game.init()

    for i in range(10):
        game.new_episode()
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        while not game.is_episode_finished():
            action_probability_distribution = sess.run(PGNet.action_distribution, feed_dict={
                PGNet.inputs_: state.reshape(1, *state_size)
            })
            action = np.random.choice(range(action_probability_distribution.shape[1]),
                                      p=action_probability_distribution.ravel())
            action = possible_actions[action]
            reward = game.make_action(action)
            done = game.is_episode_finished()
            if done:
                break
            else:
                next_state = game.get_state().screen_buffer
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                state = next_state

        print("Score for episode ", i, " :", game.get_total_reward())

    game.close()
