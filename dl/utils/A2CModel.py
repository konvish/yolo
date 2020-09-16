#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by kong on 2020/9/1

import tensorflow as tf
import numpy as np
from baselines.common import explained_variance
from baselines.common.atari_wrappers import LazyFrames
from baselines.common.runners import AbstractEnvRunner
from baselines.common.vec_env import SubprocVecEnv
from baselines import logger
import time
from dl.utils.utilities import find_trainable_variables, mse
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv


class A2CModel(object):
    """
    A2C 模型 \n
    __init__:
    -创建step_model
    -创建train_model

    train():
    -训练模型（feedforward，backward)

    save/load():
    -保存/加载模型
    """

    def __init__(self, policy, ob_space, action_space, nenvs, nsteps, ent_coef, vf_coef, max_grad_norm):
        """
        初始化
        :param policy:a2c policy
        :param ob_space: 状态
        :param action_space: 动作
        :param nenvs: 多少个环境
        :param nsteps: 多少步伐(连续多少个步伐进行一次训练)
        :param ent_coef: entropy coef
        :param vf_coef: value coef
        :param max_grad_norm:max grad norm
        """
        sess = tf.get_default_session()
        actions_ = tf.placeholder(tf.int32, [None], name="actions_")
        advantages_ = tf.placeholder(tf.float32, [None], name="advantages_")
        rewards_ = tf.placeholder(tf.float32, [None], name="rewards_")
        lr_ = tf.placeholder(tf.float32, name="learning_rate_")
        # step model train model
        step_model = policy(sess, ob_space, action_space, nenvs, 1, reuse=False)
        train_model = policy(sess, ob_space, action_space, nenvs * nsteps, nsteps, reuse=True)
        """
        计算loss
        loss = policy gradient loss - entropy * entropy coefficient + value coefficient * value loss
        """
        # policy loss
        # output - log(pi)
        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=actions_)

        # 1/n * sum A(si,ai) * -logpi(ai|si)
        pg_loss = tf.reduce_mean(advantages_ * neglogpac)

        # value loss
        # 1/2 * sum[R-V(s)]^2
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), rewards_))

        # 利用熵将收敛限制为次优策略，以提高搜索效率。
        entropy = tf.reduce_mean(train_model.pd.entropy())
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        """
        计算loss的梯度并更新参数
        """
        # 1.获取模型参数
        params = find_trainable_variables("model")
        # 2.计算梯度
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        # 3.创建优化器
        trainer = tf.train.RMSPropOptimizer(learning_rate=lr_, decay=0.99, epsilon=1e-5)
        # 4.bp
        _train = trainer.apply_gradients(grads)

        def train(state_in, actions, returns, values, lr):
            """
            advantage：A(s,a) = R + yV(s') - V(s)
            returns = R + yV(s')
            计算策略的loss，价值的loss，策略的entropy
            :param state_in: 状态
            :param actions: 动作
            :param returns: 反馈
            :param values: 价值
            :param lr: 训练率
            :return: policy loss,value loss,policy entropy
            """
            advantages = returns - values
            td_map = {train_model.inputs_: state_in, actions_: actions, advantages_: advantages
                , rewards_: returns, lr_: lr}
            policy_loss, value_loss, policy_entropy, _ = sess.run([pg_loss, vf_loss, entropy, _train], td_map)
            return policy_loss, value_loss, policy_entropy

        def save(save_path):
            """
            保存模型
            :param save_path:保存路径
            :return: none
            """
            saver = tf.train.Saver()
            saver.save(sess, save_path)

        def load(load_path):
            """
            加载模型
            :param load_path:模型路径
            :return: model
            """
            saver = tf.train.Saver()
            print('Loading ' + load_path)
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
    """
    创建mini batch经验
    """

    def __init__(self, env, model, nsteps, total_timesteps, gamma, lam):
        """
        初始化
        :param env:环境
        :param model: 模型
        :param nsteps: 步伐
        :param total_timesteps:总步伐
        :param gamma: 衰减系数
        :param lam: GAE（General Advantage Estimation）
        """
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.gamma = gamma
        self.lam = lam
        self.total_timesteps = total_timesteps

    def run(self):
        """
        创建mini batch经验
        :return:
        """
        mb_obs, mb_actions, mb_rewards, mb_values, mb_dones = [], [], [], [], []
        for n in range(self.nsteps):
            # 通过obs获取action，value
            actions, values = self.model.step(self.obs, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
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
        mb_actions = np.asarray(mb_actions, dtype=np.int32)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs)  # 最后一个状态的价值

        mb_returns = np.zeros_like(mb_rewards)
        mb_advantages = np.zeros_like(mb_rewards)
        lastgaelam = 0

        # 从最后往前计算价值
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
        return map(sf01, (mb_obs, mb_actions, mb_returns, mb_values))


def sf01(arr):
    """
    交换并flatten
    :param arr: shape
    :return:
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def learn(policy, env, nsteps, total_timesteps, gamma, lam, vf_coef, ent_coef, lr, max_grad_norm, log_interval,
          name='sonic', nenvs=1, update=-1):
    """
    训练模型
    :param policy:模型策略
    :param env: 环境
    :param nsteps: 步伐
    :param total_timesteps:总步伐
    :param gamma: 衰减率
    :param lam: gae
    :param vf_coef:value coef
    :param ent_coef: entropy coef
    :param lr: 学习率
    :param max_grad_norm:max grad norm
    :param log_interval: 每隔多少保存一次模型
    :return:
    """
    noptepochs = 4
    nminibatches = 8
    ob_space = env.observation_space
    ac_space = env.action_space
    batch_size = nenvs * nsteps
    batch_train_size = batch_size // nminibatches
    assert batch_size % nminibatches == 0
    model = A2CModel(policy=policy, ob_space=ob_space, action_space=ac_space, nenvs=nenvs
                     , nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm)
    if update > -1:
        load_path = "./models/" + str(update) + "/sonic-a2c.ckpt"
        model.load(load_path)
    runner = Runner(env, model, nsteps, total_timesteps, gamma, lam)
    tfirststart = time.time()
    model_count = update

    for update in range(1, total_timesteps // batch_size + 1):
        tstart = time.time()
        obs, actions, returns, values = runner.run()
        mb_losses = []
        total_batches_train = 0
        indices = np.arange(batch_size)
        for _ in range(noptepochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, batch_train_size):
                end = start + batch_train_size
                mbinds = indices[start:end]
                slices = (arr[mbinds] for arr in (obs, actions, returns, values))
                mb_losses.append(model.train(*slices, lr))
        # feedforward -> get losses -> update
        lossvalues = np.mean(mb_losses, axis=0)
        tnow = time.time()
        fps = int(batch_size / (tnow - tstart))

        if update % log_interval == 0 or update == 1:
            """
            计算explains variance
            returns 1-[y-ypred] / y
            interpretation:
            ev=0 => might as well have predicted zero 
            ev=1 => prefect prediction
            ev<0 => worse than just prediction zero
            """
            ev = explained_variance(values, returns)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update * batch_size)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_loss", float(lossvalues[0]))
            logger.record_tabular("policy_entropy", float(lossvalues[2]))
            logger.record_tabular("value_loss", float(lossvalues[1]))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("time elapsed", float(tnow - tfirststart))
            logger.dump_tabular()
            # 只保存最近的20个模型
            if model_count < 20:
                model_count += 1
            else:
                model_count = 0

            savepath = "./models/" + str(model_count) + "/" + name + "-a2c.ckpt"
            model.save(savepath)
            print("Saving to", savepath)
    env.close()


def play(policy, env, update=20, name='sonic'):
    ob_space = env.observation_space
    ac_space = env.action_space

    model = A2CModel(policy=policy,
                     ob_space=ob_space,
                     action_space=ac_space,
                     nenvs=1,
                     nsteps=1,
                     ent_coef=0,
                     vf_coef=0,
                     max_grad_norm=0)
    load_path = "./models/" + str(update) + "/" + name + "-a2c.ckpt"
    model.load(load_path)
    obs = env.reset()
    score = 0
    boom = 0
    done = False
    while not done:
        boom += 1
        if type(obs) == LazyFrames:
            obs = np.copy(obs)[np.newaxis, :, :, :]
        actions, values = model.step(obs)
        if type(env) == DummyVecEnv:
            obs, rewards, done, _ = env.step(actions)
        else:
            obs, rewards, done, _ = env.step(actions[0])
        score += rewards
        env.render()
        # time.sleep(0.01)
    print("Score ", score)
    env.close()
