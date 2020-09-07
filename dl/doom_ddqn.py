#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by kong on 2020/8/28

import tensorflow as tf
import numpy as np
from vizdoom import *
import random
import time
from skimage import transform
from collections import deque
import matplotlib.pyplot as plt
import warnings
from dl.PGNetwork import DDDQNNetwork, Memory2

warnings.filterwarnings('ignore')


def create_environment():
    game = DoomGame()
    game.load_config("deadly_corridor.cfg")
    game.set_doom_scenario_path("deadly_corridor.wad")
    game.init()
    possible_actions = np.identity(7, dtype=int).tolist()
    return game, possible_actions


game, possible_actions = create_environment()


def preprocess_frame(frame):
    cropped_frame = frame[15:-5, 20:-20]
    normalized_frame = cropped_frame / 255.0
    preprocessed_frame = transform.resize(cropped_frame, [100, 120])
    return preprocessed_frame


stack_size = 4
stacked_frames = deque([np.zeros((100, 120), dtype=np.int) for i in range(stack_size)], maxlen=4)


def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)
    if is_new_episode:
        stacked_frames = deque([np.zeros((100, 120), dtype=np.int) for i in range(stack_size)], maxlen=4)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)
    else:
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)
    return stacked_state, stacked_frames


state_size = [100, 120, 4]
action_size = game.get_available_buttons_size()
learning_rate = 0.00025
total_episodes = 5000
max_steps = 5000
batch_size = 64
max_tau = 10000
explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.00005
gamma = 0.95
pretrain_length = 100000
memory_size = 100000
training = False
episode_render = False

tf.reset_default_graph()
ddqnNet = DDDQNNetwork(state_size, action_size, learning_rate)
targetNetwork = DDDQNNetwork(state_size, action_size, learning_rate, name="TargetNet")
memory = Memory2(memory_size)
game.new_episode()

for i in range(pretrain_length):
    if i == 0:
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)
    action = random.choice(possible_actions)
    reward = game.make_action(action)
    done = game.is_episode_finished()
    if done:
        next_state = np.zeros(state.shape)
        experience = state, action, reward, next_state, done
        memory.store(experience)
        game.new_episode()
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)
    else:
        next_state = game.get_state().screen_buffer
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
        experience = state, action, reward, next_state, done
        memory.store(experience)
        state = next_state

writer = tf.summary.FileWriter("/tensorboard/dddqn/1")
tf.summary.scalar("Loss", ddqnNet.loss)
write_op = tf.summary.merge_all()


def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    exp_exp_tradeoff = np.random.rand()
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    if explore_probability > exp_exp_tradeoff:
        action = random.choice(possible_actions)
    else:
        Qs = sess.run(ddqnNet.output, feed_dict={ddqnNet.inputs_: state.reshape((1, *state.shape))})
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]
    return action, explore_probability


def update_target_graph():
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DDDQNNet")
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TargetNet")
    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


saver = tf.train.Saver()
if training:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        decay_step = 0
        tau = 0
        game.init()
        update_target = update_target_graph()
        sess.run(update_target)
        for episode in range(total_episodes):
            step = 0
            episode_rewards = []
            game.new_episode()
            state = game.get_state().screen_buffer
            state, stacked_frames = stack_frames(stacked_frames, state, True)

            while step < max_steps:
                step += 1
                tau += 1
                decay_step += 1
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state,
                                                             possible_actions)
                reward = game.make_action(action)
                done = game.is_episode_finished()

                episode_rewards.append(reward)
                if done:
                    next_state = np.zeros((120, 140), dtype=np.int)
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    step = max_steps
                    total_reward = np.sum(episode_rewards)
                    print("Episode: {}".format(episode),
                          "Total reward: {}".format(total_reward),
                          "Training loss: {:.4f}".format(loss),
                          "Explore P: {:.4f}".format(explore_probability))
                    experience = state, action, reward, next_state, done
                    memory.store(experience)
                else:
                    next_state = game.get_state().screen_buffer
                    next_state, stack_frames = stack_frames(stacked_frames, next_state, False)
                    experience = state, action, reward, next_state, done
                    memory.store(experience)
                    state = next_state
                tree_idx, batch, ISWeights_mb = memory.sample(batch_size)
                states_mb = np.array([each[0][0] for each in batch], ndmin=3)
                actions_mb = np.array([each[0][1] for each in batch])
                rewards_mb = np.array([each[0][2] for each in batch])
                next_states_mb = np.array([each[0][3] for each in batch], ndmin=3)
                dones_mb = np.array([each[0][4] for each in batch])

                target_Qs_batch = []

                q_next_state = sess.run(ddqnNet.output, feed_dict={ddqnNet.inputs_: next_states_mb})
                q_target_next_state = sess.run(targetNetwork.output, feed_dict={targetNetwork.inputs_: next_states_mb})

                for i in range(0, len(batch)):
                    terminal = dones_mb[i]
                    action = np.argmax(q_next_state[i])
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])
                    else:
                        target = rewards_mb[i] + gamma * q_target_next_state[i][action]
                        target_Qs_batch.append(target)

                targets_mb = np.array([each for each in target_Qs_batch])
                _, loss, absolute_errors = sess.run([ddqnNet.optimizer, ddqnNet.loss, ddqnNet.absolute_errors],
                                                    feed_dict={
                                                        ddqnNet.inputs_: states_mb,
                                                        ddqnNet.target_Q: targets_mb,
                                                        ddqnNet.actions_: actions_mb,
                                                        ddqnNet.ISWeights_: ISWeights_mb
                                                    })
                memory.batch_update(tree_idx, absolute_errors)
                summary = sess.run(write_op, feed_dict={
                    ddqnNet.inputs_: states_mb,
                    ddqnNet.target_Q: targets_mb,
                    ddqnNet.actions_: actions_mb,
                    ddqnNet.ISWeights_: ISWeights_mb
                })
                writer.add_summary(summary, episode)
                writer.flush()

                if tau > max_tau:
                    update_target = update_target_graph()
                    sess.run(update_target)
                    tau = 0
                    print("model updated")
            if episode % 5 == 0:
                save_path = saver.save(sess, "./models/doom_ddqn.ckpt")
                print("model Saved")

with tf.Session() as sess:
    game = DoomGame()
    game.load_config("deadly_corridor_testing.cfg")
    game.set_doom_scenario_path("deadly_corridor.wad")
    game.init()

    saver.restore(sess, "./models/doom_ddqn.ckpt")
    game.init()

    for i in range(10):
        game.new_episode()
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        while not game.is_episode_finished():
            exp_exp_tradeoff = np.random.rand()

            explore_probability = 0.01
            if explore_probability > exp_exp_tradeoff:
                action = random.choice(possible_actions)
            else:
                Qs = sess.run(ddqnNet.output, feed_dict={ddqnNet.inputs_: state.reshape((1, *state.shape))})
                choice = np.argmax(Qs)
                action = possible_actions[int(choice)]
            game.make_action(action)
            done = game.is_episode_finished()

            if done:
                break
            else:
                next_state = game.get_state().screen_buffer
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                state = next_state
        score = game.get_total_reward()
        print("Score: ", score)
    game.close()
