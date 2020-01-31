#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:17:11 2019

@author: fhfonsecaa
"""
import os
import gym
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense 
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard


from gym.envs.registration import register

import utility as utils
# from utility import ProbUtils
from buffer import Buffer

class Agent:
    def __init__(self, state_dim, action_dim, env, epsilon, mujoco):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.env = env
        self.EPSILON = 0.4
        self.ALPHA = 0.95
        self.GAMMA = 0.99
        self.GAE_LAMBDA = 0.95
        self.CLIPPING_LOSS_RATIO = 0.1
        self.ENTROPY_LOSS_RATIO = 0.001
        self.isMujoco = mujoco
        
        self.actor_model = self.init_actor_network()
        self.critic_model = self.init_critic_network()
        self.actor_old_model = self.init_actor_network()
        self.critic_old_model = self.init_critic_network()
		      
        self.policy_clip = 0.2 
        self.value_clip = 0.2    
        self.entropy_coef = 0.0
        self.vf_loss_coef = 0.5
        self.minibatch = 25        
        self.ppo_epochs = 10
        self.action_std = 1.0
        self.epsilon = epsilon
        
        self.cov_mat = tf.fill((action_dim,), self.action_std * self.action_std)
        self.optimizer = Adam(learning_rate = 3e-4)
        self.buffer = Buffer()
    
    def set_epsilon(self,epsilon):
        self.epsilon = epsilon  

    def get_epsilon(self):
        return self.epsilon  

    def init_actor_network(self):
        actor_model = tf.keras.Sequential()
        actor_model.add(Dense(128, input_shape=self.state_dim, activation='relu', kernel_initializer=RandomUniform(seed=69)))
        if self.isMujoco:
            actor_model.add(Dense(128, activation='relu', kernel_initializer=RandomUniform(seed=69)))
            actor_model.add(Dense(128, activation='relu', kernel_initializer=RandomUniform(seed=69)))
            actor_model.add(Dense(128, activation='relu', kernel_initializer=RandomUniform(seed=69)))
        actor_model.add(Dense(64, activation='relu', kernel_initializer=RandomUniform(seed=69)))
        actor_model.add(Dense(self.action_dim, activation = 'tanh', kernel_initializer=RandomUniform(seed=69)))
        return actor_model
    
    def init_critic_network(self):
        critic_model = tf.keras.Sequential()
        critic_model.add(Dense(128, input_shape=self.state_dim, activation='relu', kernel_initializer=RandomUniform(seed=69)))
        if self.isMujoco:
            critic_model.add(Dense(128, activation='relu', kernel_initializer=RandomUniform(seed=69)))
            critic_model.add(Dense(128, activation='relu', kernel_initializer=RandomUniform(seed=69)))
            critic_model.add(Dense(128, activation='relu', kernel_initializer=RandomUniform(seed=69)))
        critic_model.add(Dense(64, activation='relu', kernel_initializer=RandomUniform(seed=69)))
        critic_model.add(Dense(1, activation = 'linear', kernel_initializer=RandomUniform(seed=69)))
        return critic_model

    @tf.function
    def get_action(self, state):        
        state = tf.expand_dims(tf.cast(state, dtype = tf.float32), 0)        
        action_mean = self.actor_model(state)
        action = utils.normal_sample(action_mean, self.cov_mat)
        return tf.squeeze(action)   

    def save_eps(self, state, reward, action, done, next_state):
        self.buffer.save_eps(state, reward, action, done, next_state)

    def get_ppo_loss(self, states, actions, rewards, dones, next_states):        
        action_mean, values  = self.actor_model(states), self.critic_model(states)
        old_action_mean, old_values = self.actor_old_model(states), self.critic_old_model(states)
        next_values  = self.critic_model(next_states)  
                
        # Block the contribution of the old value in backpropagation
        old_values = tf.stop_gradient(old_values)

        # Calculate entropy of the action probability 
        dist_entropy = tf.math.reduce_mean(utils.entropy(action_mean, self.cov_mat))

        # Calculate the ratio (pi_theta / pi_theta__old)        
        logprobs = tf.expand_dims(utils.logprob(action_mean, self.cov_mat, actions), 1)         
        old_logprobs = tf.stop_gradient(tf.expand_dims(utils.logprob(old_action_mean, self.cov_mat, actions), 1))
		
        # Calculate external GAE
        advantages = tf.stop_gradient(utils.generalized_advantage_estimation(values, rewards, next_values, dones))
        returns = tf.stop_gradient(utils.temporal_difference(rewards, next_values, dones))
                        
        # Calculate external critic loss through clipped critic value
        vpredclipped = old_values + tf.clip_by_value(values - old_values, -self.value_clip, self.value_clip) # Minimize the difference between old value and new value
        vf_losses1 = tf.math.square(returns - values) # Mean Squared Error
        vf_losses2 = tf.math.square(returns - vpredclipped) # Mean Squared Error        
        critic_loss = tf.math.reduce_mean(tf.math.maximum(vf_losses1, vf_losses2)) * 0.5        
        
        # Calculate Surrogate Loss
        ratios = tf.math.exp(logprobs - old_logprobs) # ratios = old_logprobs / logprobs        
        surr1 = ratios * advantages
        surr2 = tf.clip_by_value(ratios, 1 - self.policy_clip, 1 + self.policy_clip) * advantages
        pg_loss = tf.math.reduce_mean(tf.math.minimum(surr1, surr2))         
                        
        loss = (critic_loss * self.vf_loss_coef) - (dist_entropy * self.entropy_coef) - pg_loss
        return loss       
    
    # Get loss and perform backpropagation
    @tf.function
    def train_ppo(self, states, actions, rewards, dones, next_states):     
        with tf.GradientTape() as ppo_tape:
            loss = self.get_ppo_loss(states, actions, rewards, dones, next_states)
                    
        gradients = ppo_tape.gradient(loss, self.actor_model.trainable_variables + self.critic_model.trainable_variables)        
        self.optimizer.apply_gradients(zip(gradients, self.actor_model.trainable_variables + self.critic_model.trainable_variables)) 
        
    # Update the model
    def update_ppo(self):        
        batch_size = int(self.buffer.length() / self.minibatch)
        
        # Optimize policy using K epochs:
        for _ in range(self.ppo_epochs):       
            for states, actions, rewards, dones, next_states in self.buffer.get_all().batch(batch_size):
                self.train_ppo(states, actions, rewards, dones, next_states)
                    
        self.buffer.clean_buffer()
                
        # Copy new weights into old policy:
        self.actor_old_model.set_weights(self.actor_model.get_weights())
        self.critic_old_model.set_weights(self.critic_model.get_weights())

    def save_weights(self,episode,identifier):
        env_name = self.env.unwrapped.spec.id
        time = utils.get_time_date()
        if os.path.exists(env_name) is False:
            os.mkdir(env_name)
        print('Saving weights as -{}- {}'.format(identifier,time))
        self.actor_model.save_weights(env_name+'/actor_weights_'+str(episode)+identifier+'.hd5')
        self.actor_old_model.save_weights(env_name+'/actor_old_weights_'+str(episode)+identifier+'.hd5')
        self.critic_model.save_weights(env_name+'/critic_weights_'+str(episode)+identifier+'.hd5')
        self.critic_old_model.save_weights(env_name+'/critic_old_weights_'+str(episode)+identifier+'.hd5')

    def save_models(self,episode,identifier):
        env_name = self.env.unwrapped.spec.id
        time = utils.get_time_date()
        if os.path.exists(env_name) is False:
            os.mkdir(env_name)
        print('Saving models as -{}- {}'.format(identifier,time))
        self.actor_model.save(env_name+'/actor_model_'+str(episode)+identifier+time+'.h5')
        self.actor_old_model.save(env_name+'/actor_old_model_'+str(episode)+identifier+time+'.h5')
        self.critic_model.save(env_name+'/critic_model_'+str(episode)+identifier+time+'.h5')
        self.critic_old_model.save(env_name+'/critic_old_model_'+str(episode)+identifier+time+'.h5')