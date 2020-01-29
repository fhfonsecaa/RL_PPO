#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 19:47:41 2019

@author: fhfonsecaa
"""
import gym
from gym.envs.registration import register
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

tf.random.set_seed(69)

class ProbUtils:
    def __init__(self):
        self.gamma = 0.99
        self.lam = 0.95
    
    def normal_sample(self, mean, cov_mat):
        dist = tfp.distributions.MultivariateNormalDiag(mean, cov_mat)
        return dist.sample()
        
    def entropy(self, mean, cov_mat):
        dist = tfp.distributions.MultivariateNormalDiag(mean, cov_mat)            
        return dist.entropy()
      
    def logprob(self, mean, cov_mat, value_data):
        dist = tfp.distributions.MultivariateNormalDiag(mean, cov_mat)
        return dist.log_prob(value_data)
      
    def normalize(self, data):
        data_normalized = (data - tf.math.reduce_mean(data)) / (tf.math.reduce_std(data) + 1e-8)
        return data_normalized     
      
    def temporal_difference(self, rewards, next_values, dones):
        temporal_diff = rewards + self.gamma * next_values * (1 - dones)        
        return temporal_diff
      
    def generalized_advantage_estimation(self, values, rewards, next_value, done):
        gae = 0
        returns = []
        
        for step in reversed(range(rewards.shape[0])):   
            delta = rewards[step] + self.gamma * next_value[step] * (1 - done[step]) - values[step]
            gae = delta + (self.lam * gae)
            returns.insert(0, gae)
            
        return tf.stack(returns)