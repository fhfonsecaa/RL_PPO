#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 19:47:41 2019

@author: fhfonsecaa
"""

import gym
env = gym.make('Humanoid-v2')
env.reset()
name = env.unwrapped.spec.id
print(name)
for _ in range(1000):
  env.render()
  env.step(env.action_space.sample()) # take a random action