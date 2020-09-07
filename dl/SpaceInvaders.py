#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by kong on 2020/8/28

import tensorflow as tf
import numpy as np
import retro
from skimage import transform
from skimage.color import rgb2gray

import matplotlib.pyplot as plt
from collections import deque
import random
import warnings
from dl.PGNetwork import DQNetwork, Memory

warnings.filterwarnings("ignore")

env = retro.make(game='SpaceInvaders-Atari2600')
print("The size of out frame is: ", env.observation_space)
print("The action size is : ", env.action_space.n)
possible_actions = np.array(np.identity(env.action_space.n, dtype=int).tolist())


def preprocess_frame(frame):
    gray = rgb2gray(frame)
    cropped_frame = gray[8:-12, 4:-12]
    normalized_frame = cropped_frame / 255.0
    preprocessed_frame = transform.resize(normalized_frame, [110, 84])
    return preprocessed_frame


stack_size = 4
stacked_frames = deque([np.zeros((110, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)


def stack_frame(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)
    if is_new_episode:
        stacked_frames = deque([np.zeros((110, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)
    else:
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)
    return stacked_state, stacked_frames


state_size = [110, 84, 4]
action_size = env.action_space.n
learning_rate = 0.00025
total_episodes = 50
max_steps = 50000
batch_size = 64
explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.00001
gamma = 0.9
pretrain_length = batch_size
memory_size = 1000000
stack_size = 4
training = False
episode_render = False

tf.reset_default_graph()
DQNet = DQNetwork(state_size, action_size, learning_rate)
memory = Memory(max_size=memory_size)
for i in range(pretrain_length):
    if i == 0:
        state = env.reset()
        state, stacked_frames = stack_frame(stacked_frames, state, True)

    choice = random.randint(1, len(possible_actions)) - 1
    action = possible_actions[choice]
    next_state, reward, done, _ = env.step(action)
    next_state, stacked_frames = stack_frame(stacked_frames, next_state, False)
    if done:
        next_state = np.zeros(state.shape)
        memory.add((state, action, reward, next_state, done))
        state = env.reset()
        state, stacked_frames = stack_frame(stacked_frames, state, True)
    else:
        memory.add((state, action, reward, next_state, done))
        state = next_state

writer = tf.summary.FileWriter("/tensorboard/dqn/1")
tf.summary.scalar("Loss", DQNet.loss)
write_op = tf.summary.merge_all()


def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    exp_exp_tradeoff = np.random.rand()
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    if explore_probability > exp_exp_tradeoff:
        choice = random.randint(1, len(possible_actions)) - 1
        action = possible_actions[choice]
    else:
        Qs = sess.run(DQNet.output, feed_dict={DQNet.inputs_: state.reshape((1, *state.shape))})
        choice = np.argmax(Qs)
        action = possible_actions[choice]
    return action, explore_probability


saver = tf.train.Saver()
if training:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        decay_step = 0
        for episode in range(total_episodes):
            step = 0
            episode_rewards = []
            state = env.reset()
            state, stacked_frames = stack_frame(stacked_frames, state, True)
            while step < max_steps:
                step += 1
                decay_step += 1
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state,
                                                             possible_actions)
                next_state, reward, done, _ = env.step(action)
                if episode_render:
                    env.render()
                episode_rewards.append(reward)
                if done:
                    next_state = np.zeros((110, 84), dtype=np.int)
                    next_state, stacked_frames = stack_frame(stacked_frames, next_state, False)
                    step = max_steps
                    total_reward = np.sum(episode_rewards)
                    print("Episode: {}".format(episode), "Total reward: {}".format(total_reward),
                          "Explore P: {:.4f}".format(explore_probability), "Training Loss {:.4f}".format(loss))
                    # rewards_list.append((episode, total_reward))
                    memory.add((state, action, reward, next_state, done))
                else:
                    next_state, stacked_frames = stack_frame(stacked_frames, next_state, False)
                    memory.add((state, action, reward, next_state, done))
                    state = next_state
                batch = memory.sample(batch_size)
                states_mb = np.array([each[0] for each in batch], ndmin=3)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch])
                next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                dones_mb = np.array([each[4] for each in batch])

                target_Qs_batch = []
                Qs_next_state = sess.run(DQNet.output, feed_dict={
                    DQNet.inputs_: next_states_mb
                })

                for i in range(0, len(batch)):
                    terminal = dones_mb[i]
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])
                    else:
                        target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                        target_Qs_batch.append(target)
                targets_mb = np.array([each for each in target_Qs_batch])
                loss, _ = sess.run([DQNet.loss, DQNet.optimizer], feed_dict={
                    DQNet.inputs_: states_mb,
                    DQNet.target_Q: targets_mb,
                    DQNet.actions_: actions_mb
                })
                summary = sess.run(write_op, feed_dict={
                    DQNet.inputs_: states_mb,
                    DQNet.target_Q: targets_mb,
                    DQNet.actions_: actions_mb
                })
                writer.add_summary(summary, episode)
                writer.flush()
            if episode % 5 == 0:
                save_path = saver.save(sess, "./models/space_invaders.ckpt")
                print("Model Saved")

with tf.Session() as sess:
    total_test_rewards = []
    saver.restore(sess, "./models/space_invaders.ckpt")

    for episode in range(1):
        total_rewards = 0
        state = env.reset()
        state, stacked_frames = stack_frame(stacked_frames, state, True)

        print("***********************************")
        print("EPISODE", episode)

        while True:
            state = state.reshape((1, *state_size))
            Qs = sess.run(DQNet.output, feed_dict={DQNet.inputs_: state})
            choice = np.argmax(Qs)
            action = possible_actions[choice]

            next_state, reward, done, _ = env.step(action)
            env.render()

            total_rewards += reward
            if done:
                print("Score", total_rewards)
                total_test_rewards.append(total_rewards)
                break
            next_state, stacked_frames = stack_frame(stacked_frames, next_state, False)
            state = next_state
    env.close()
