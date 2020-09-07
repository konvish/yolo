#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by kong on 2020/9/2
import retro


def test():
    # AdventureIslandII-Nes,AdventureIsland3-Nes
    env = retro.make(game="AdventureIslandII-Nes")
    obs = env.reset()
    print(env.action_space.n)
    # while True:
    #     obs, rew, done, info = env.step(env.action_space.sample())
    #     env.render()
    #     if done:
    #         obs = env.reset()
    env.close()


if __name__ == '__main__':
    test()
