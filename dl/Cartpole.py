#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by kong on 2020/8/26

import tensorflow as tf
import numpy as np
import gym

env = gym.make("CartPole-v0")
env = env.unwrapped
env.seed(1)

state_size = 4
action_size = env.action_space.n  # 左右
max_episodes = 300
learning_rate = 0.01
gamma = 0.95


def discount_and_normalize_rewards(episode_rewards):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative

    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - mean) / std  # 归一化
    return discounted_episode_rewards


with tf.name_scope("inputs"):
    input_ = tf.placeholder(tf.float32, [None, state_size], name="input_")
    actions = tf.placeholder(tf.int32, [None, action_size], name="actions")
    discounted_episode_rewards_ = tf.placeholder(tf.float32, [None, ], name="discounted_episode_rewards_")

    mean_reward_ = tf.placeholder(tf.float32, name="mean_reward")

    with tf.name_scope("fc1"):
        fc1 = tf.contrib.layers.fully_connected(inputs=input_,
                                                num_outputs=10,
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())

    with tf.name_scope("fc2"):
        fc2 = tf.contrib.layers.fully_connected(inputs=fc1,
                                                num_outputs=action_size,
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())
    with tf.name_scope("fc3"):
        fc3 = tf.contrib.layers.fully_connected(inputs=fc2,
                                                num_outputs=action_size,
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())

    with tf.name_scope("softmax"):
        action_distribution = tf.nn.softmax(fc3)

    with tf.name_scope("loss"):
        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc3, labels=actions)
        loss = tf.reduce_mean(neg_log_prob * discounted_episode_rewards_)

    with tf.name_scope("train"):
        train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

writer = tf.summary.FileWriter("/tensorboard/pg/1")
tf.summary.scalar("Loss", loss)
tf.summary.scalar("Reward_mean", mean_reward_)
write_op = tf.summary.merge_all()

allRewards = []
total_rewards = 0
maximumRewardRecorded = 0
episode = 0
episode_states, episode_actions, episode_rewards = [], [], []
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for episode in range(max_episodes):
        episode_rewards_sum = 0
        state = env.reset()
        env.render()

        while True:
            action_probability_distribution = sess.run(action_distribution, feed_dict={input_: state.reshape([1, 4])})
            action = np.random.choice(range(action_probability_distribution.shape[1]),
                                      p=action_probability_distribution.ravel())

            new_state, reward, done, info = env.step(action)
            episode_states.append(state)
            action_ = np.zeros(action_size)
            action_[action] = 1
            episode_actions.append(action_)
            episode_rewards.append(reward)
            if done:
                episode_rewards_sum = np.sum(episode_rewards)
                allRewards.append(episode_rewards_sum)
                total_rewards = np.sum(allRewards)
                mean_reward = np.divide(total_rewards, episode + 1)

                maximumRewardRecorded = np.amax(allRewards)
                print("++++++++++++++++++++++++++++++++++")
                print("Episode: ", episode)
                print("Reward: ", episode_rewards_sum)
                print("Mean Reward: ", mean_reward)
                print("Max reward so far: ", maximumRewardRecorded)

                discounted_episode_rewards = discount_and_normalize_rewards(episode_rewards)
                loss_, _ = sess.run([loss, train_opt], feed_dict={input_: np.vstack(np.array(episode_states)),
                                                                  actions: np.vstack(np.array(episode_actions)),
                                                                  discounted_episode_rewards_: discounted_episode_rewards
                                                                  })

                summary = sess.run(write_op, feed_dict={
                    input_: np.vstack(np.array(episode_states)),
                    actions: np.vstack(np.array(episode_actions)),
                    discounted_episode_rewards_: discounted_episode_rewards,
                    mean_reward_: mean_reward
                })

                writer.add_summary(summary, episode)
                writer.flush()

                episode_states, episode_actions, episode_rewards = [], [], []
                break
            state = new_state
        if episode % 100 == 0:
            saver.save(sess, "./models/model.ckpt")
            print("Model saved")

with tf.Session() as sess:
    env.reset()
    rewards = []

    saver.restore(sess, "./models/model.ckpt")

    for episode in range(10):
        state = env.reset()
        step = 0
        done = False
        total_rewards = 0
        print("++++++++++++++++++++++++")
        print("EPISODE", episode)

        while True:
            action_probability_distribution = sess.run(action_distribution, feed_dict={input_: state.reshape([1, 4])})
            action = np.random.choice(range(action_probability_distribution.shape[1]),
                                      p=action_probability_distribution.ravel())

            new_state, reward, done, info = env.step(action)
            total_rewards += reward

            if done:
                rewards.append(total_rewards)
                print("Score", total_rewards)
                break
            state = new_state
    env.close()
    print("Score over time: " + str(sum(rewards) / 10))
