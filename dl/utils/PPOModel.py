#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by kong on 2020/9/2

import tensorflow as tf
from dl.utils.utilities import find_trainable_variables
import numpy as np
import time
from baselines import logger

from baselines.common.runners import AbstractEnvRunner
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import LazyFrames
from baselines.common import explained_variance


class PPOModel(object):
    def __init__(self, policy, ob_space, action_space, nenvs, nsteps, ent_coef, vf_coef, max_grad_norm):
        sess = tf.get_default_session()
        actions_ = tf.placeholder(tf.int32, [None], name="actions_")
        advantages_ = tf.placeholder(tf.float32, [None], name="advantages_")
        rewards_ = tf.placeholder(tf.float32, [None], name="rewards_")
        lr_ = tf.placeholder(tf.float32, name="learning_rate_")
        oldneglopac_ = tf.placeholder(tf.float32, [None], name="oldneglopac_")
        oldvpred_ = tf.placeholder(tf.float32, [None], name="oldvpred_")
        cliprange_ = tf.placeholder(tf.float32, [])
        step_model = policy(sess, ob_space, action_space, nenvs, 1, reuse=False)
        train_model = policy(sess, ob_space, action_space, nenvs * nsteps, nsteps, reuse=True)
        value_prediction = train_model.vf
        value_prediction_clipped = oldvpred_ + tf.clip_by_value(train_model.vf - oldvpred_, -cliprange_, cliprange_)
        value_loss_unclipped = tf.square(value_prediction - rewards_)
        value_loss_clipped = tf.square(value_prediction_clipped - rewards_)
        vf_loss = 0.5 * tf.reduce_mean(tf.maximum(value_loss_unclipped, value_loss_clipped))
        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=actions_)
        ratio = tf.exp(oldneglopac_ - neglogpac)
        pg_loss_unclipped = -advantages_ * ratio
        pg_loss_clipped = -advantages_ * tf.clip_by_value(ratio, 1.0 - cliprange_, 1.0 + cliprange_)
        pg_loss = tf.reduce_mean(tf.maximum(pg_loss_unclipped, pg_loss_clipped))
        entropy = tf.reduce_mean(train_model.pd.entropy())
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))

        trainer = tf.train.RMSPropOptimizer(learning_rate=lr_, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)

        def train(state_in, actions, returns, values, neglogpacs, lr, cliprange):
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            td_map = {train_model.inputs_: state_in, actions_: actions,
                      advantages_: advantages,
                      rewards_: returns,
                      lr_: lr,
                      cliprange_: cliprange,
                      oldneglopac_: neglogpacs,
                      oldvpred_: values}
            policy_loss, value_loss, policy_entropy, _ = sess.run([pg_loss, vf_loss, entropy, _train], td_map)
            return policy_loss, value_loss, policy_entropy

        def save(save_path):
            saver = tf.train.Saver()
            saver.save(sess, save_path)

        def load(load_path):
            saver = tf.train.Saver()
            saver.restore(sess, load_path)

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)


class Runner(AbstractEnvRunner):
    def __init__(self, env, model, nsteps, total_timesteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.gamma = gamma
        self.lam = lam
        self.total_timesteps = total_timesteps

    def run(self):
        mb_obs, mb_actions, mb_rewards, mb_values, mb_neglopacs, mb_dones = [], [], [], [], [], []
        for n in range(self.nsteps):
            actions, values, neglopacs = self.model.step(self.obs, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglopacs.append(neglopacs)
            mb_dones.append(self.dones)
            if type(self.env) == SubprocVecEnv:
                self.obs[:], rewards, self.dones, _ = self.env.step(actions)
            else:
                ob, reward, done, _ = self.env.step(actions[0])
                self.obs = np.array([ob])
                rewards = np.array([reward])
                self.dones = np.array([done])
            mb_rewards.append(rewards)

        mb_obs = np.asarray(mb_obs, dtype=np.uint8)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions, dtype=np.float32)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglopacs = np.asarray(mb_neglopacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs)
        mb_returns = np.zeros_like(mb_rewards)
        mb_advantages = np.zeros_like(mb_rewards)

        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t + 1]
                nextvalues = mb_values[t + 1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advantages[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advantages + mb_values
        return map(sf01, (mb_obs, mb_actions, mb_returns, mb_values, mb_neglopacs))


def sf01(arr):
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def constfn(val):
    def f(_):
        return val

    return f


def learn(policy, env, nsteps, total_timesteps, gamma, lam, vf_coef, ent_coef, lr, cliprange, max_grad_norm,
          log_interval, name='sonic', nenvs=1, update=-1, test_env=None):
    """
    学习模型
    :param policy:模型策略
    :param env: 环境
    :param nsteps: 步长，即多少步伐更新一次模型
    :param total_timesteps:总步伐
    :param gamma: 衰减系数
    :param lam: lam
    :param vf_coef:coef
    :param ent_coef: coef
    :param lr: 学习率
    :param cliprange:cliprange
    :param max_grad_norm: max grad norm
    :param log_interval: 多少次保存一次模型
    :param name: 模型名称
    :param update: update >-1,加载update模型继续学习，-1则从0开始
    :return:
    """
    noptepochs = 4
    nminibatches = 8
    if isinstance(lr, float):
        lr = constfn(lr)
    else:
        assert callable(lr)
    ob_space = env.observation_space
    ac_space = env.action_space
    batch_size = nenvs * nsteps
    batch_train_size = batch_size // nminibatches
    assert batch_size % nminibatches == 0

    model = PPOModel(policy=policy, ob_space=ob_space, action_space=ac_space, nenvs=nenvs, nsteps=nsteps,
                     ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm)
    runner = Runner(env, model, nsteps=nsteps, total_timesteps=total_timesteps, gamma=gamma, lam=lam)

    if update > -1:
        load_path = "./models/" + str(update) + "/sonic-ppo.ckpt"
        print("load model:" + load_path)
        model.load(load_path)

    tfirststart = time.time()
    nupdates = total_timesteps // batch_size + 1
    model_count = update

    for update in range(1, nupdates + 1):
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)
        obs, actions, returns, values, neglogpacs = runner.run()
        mb_losses = []
        total_batches_train = 0
        indices = np.arange(batch_size)
        for _ in range(noptepochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, batch_train_size):
                end = start + batch_train_size
                mbinds = indices[start:end]
                slices = (arr[mbinds] for arr in (obs, actions, returns, values, neglogpacs))
                mb_losses.append(model.train(*slices, lrnow, cliprangenow))

        lossvalues = np.mean(mb_losses, axis=0)
        tnow = time.time()
        fps = int(batch_size / (tnow - tstart))

        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, returns)
            logger.record_tabular("serial_timesteps", update * nsteps)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update * batch_size)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_loss", float(lossvalues[0]))
            logger.record_tabular("policy_entropy", float(lossvalues[2]))
            logger.record_tabular("value_loss", float(lossvalues[1]))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("time elapsed", float(tnow - tfirststart))

            # 只保存最近的20个模型
            if model_count < 20:
                model_count += 1
            else:
                model_count = 0

            savepath = "./models/" + str(model_count) + "/" + name + "-ppo.ckpt"

            model.save(savepath)
            print("Saving to", savepath)
            # test_score = testing(model, test_env)
            #
            # logger.record_tabular("Mean score test level", test_score)
            logger.dump_tabular()
    env.close()


def testing(model, test_env):
    """
    测试模型的分数
    :param model: 模型
    :return: score
    """
    total_score = 0
    if test_env == None:
        return total_score

    for trial in range(3):
        obs = test_env.reset()
        done = False
        score = 0
        while not done:
            action, value, _ = model.step(obs)
            obs, reward, done, info = test_env.step(action)
            score += reward[0]
        total_score += score
        trial += 1
    test_env.close()
    total_test_score = total_score / 3
    return total_test_score


def safemean(xs):
    """
    计算平均值
    :param xs:xs分数
    :return: mean
    """
    return np.nan if len(xs) == 0 else np.mean(xs)


def generate_output(policy, test_env, name='sonic'):
    """
    测试环境的反馈
    :param policy:策略
    :param test_env:测试环境
    :param name: 模型名称
    :return: score
    """
    ob_space = test_env.observation_space
    ac_space = test_env.action_space
    test_score = []
    models_indexes = [1, 6, 11, 16, 20]
    validation_model = PPOModel(policy=policy, ob_space=ob_space, action_space=ac_space,
                                nenvs=1, nsteps=1, ent_coef=0, vf_coef=0, max_grad_norm=0)
    obs = test_env.reset()
    for model_index in models_indexes:
        load_path = "./models/" + str(model_index) + "/" + name + "-ppo.ckpt"
        validation_model.load(load_path)
        score = 0
        timesteps = 0
        while timesteps < 5000:
            timesteps += 1
            actions, values, _ = validation_model.step(obs)
            obs, rewards, dones, infos = test_env.step(actions)
            score += rewards

        total_score = score / test_env.num_envs
        test_score.append(total_score)
    test_env.close()
    return test_score


def play(policy, env, update, name='sonic'):
    """
    模型可视化操作游戏
    :param policy: 模型
    :param env: 环境
    :param update: 模型版本
    :param name: 模型名称
    :return: none
    """
    ob_space = env.observation_space
    ac_space = env.action_space
    model = PPOModel(policy=policy, ob_space=ob_space, action_space=ac_space,
                     nenvs=1, nsteps=1, ent_coef=0, vf_coef=0, max_grad_norm=0)
    load_path = "./models/" + str(update) + "/" + name + "-ppo.ckpt"
    print(load_path)
    obs = env.reset()
    score = 0
    done = False

    while not done:
        if type(obs) == LazyFrames:
            obs = np.copy(obs)[np.newaxis, :, :, :]
        actions, values, _ = model.step(obs)
        if type(env) == DummyVecEnv:
            obs, rewards, done, _ = env.step(actions)
        else:
            obs, rewards, done, _ = env.step(actions[0])
        score += rewards
        env.render()
        time.sleep(0.01)
    print("Score ", score)
    env.close()
