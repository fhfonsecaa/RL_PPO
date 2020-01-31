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
from buffer import Buffer

class PPOAgent:
    def __init__(self, state_dim, action_dim, env, epsilon, mujoco):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.env = env
        self.env_name = self.env.unwrapped.spec.id
        self.epsilon = 0.4
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

        self.actor_loss_metric = tf.keras.metrics.Mean(name='actor_loss_metric')
        self.critic_loss_metric = tf.keras.metrics.Mean(name='critic_loss_metric')
        self.entropy_metric = tf.keras.metrics.Mean(name='entropy_metric')
        self.advantages_metric = tf.keras.metrics.Mean(name='advanteges_metric')
        self.returns_metric = tf.keras.metrics.Mean(name='returns_metric')
        # self.rewards_metric = tf.keras.metrics.Mean(name='rewards_metric')
        
        # tensorboard --logdir logs/gradient_tape
        self.train_log_dir = 'logs/gradient_tape/'+ self.env_name + utils.get_time_date() + '/train'
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
    
    def set_epsilon(self,epsilon):
        self.epsilon = epsilon  

    def get_epsilon(self):
        return self.epsilon  

    def get_summary_writer(self):
        return self.train_summary_writer  

    def init_actor_network(self):
        actor_model = tf.keras.Sequential()
        if self.isMujoco:
            actor_model.add(Dense(512, input_shape=self.state_dim, activation='relu', kernel_initializer=RandomUniform(seed=69)))
            actor_model.add(Dense(256, activation='relu', kernel_initializer=RandomUniform(seed=69)))
            actor_model.add(Dense(256, activation='relu', kernel_initializer=RandomUniform(seed=69)))
            actor_model.add(Dense(128, activation='relu', kernel_initializer=RandomUniform(seed=69)))
        else:
            actor_model.add(Dense(128, input_shape=self.state_dim, activation='relu', kernel_initializer=RandomUniform(seed=69)))
        actor_model.add(Dense(64, activation='relu', kernel_initializer=RandomUniform(seed=69)))
        actor_model.add(Dense(self.action_dim, activation = 'tanh', kernel_initializer=RandomUniform(seed=69)))
        return actor_model
    
    def init_critic_network(self):
        critic_model = tf.keras.Sequential()
        if self.isMujoco:
            critic_model.add(Dense(512, input_shape=self.state_dim, activation='relu', kernel_initializer=RandomUniform(seed=69)))
            critic_model.add(Dense(256, activation='relu', kernel_initializer=RandomUniform(seed=69)))
            critic_model.add(Dense(256, activation='relu', kernel_initializer=RandomUniform(seed=69)))
            critic_model.add(Dense(128, activation='relu', kernel_initializer=RandomUniform(seed=69)))
        else:
            critic_model.add(Dense(128, input_shape=self.state_dim, activation='relu', kernel_initializer=RandomUniform(seed=69)))
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
        return loss, critic_loss, dist_entropy, advantages, returns
    
    # Get loss and perform backpropagation
    @tf.function
    def train_ppo(self, states, actions, rewards, dones, next_states):     
        with tf.GradientTape() as ppo_tape:
            loss, critic_loss, dist_entropy, advantages, returns = self.get_ppo_loss(states, actions, rewards, dones, next_states)
                    
        gradients = ppo_tape.gradient(loss, self.actor_model.trainable_variables + self.critic_model.trainable_variables)        

        self.optimizer.apply_gradients(zip(gradients, self.actor_model.trainable_variables + self.critic_model.trainable_variables)) 
    
        self.actor_loss_metric(loss)
        self.critic_loss_metric(critic_loss)
        self.entropy_metric(dist_entropy)
        self.advantages_metric(advantages)
        self.returns_metric(returns)
    
    # Update the model
    def update_ppo(self):        
        batch_size = int(self.buffer.length() / self.minibatch)
        
        # Optimize policy using K epochs:
        for epoch in range(self.ppo_epochs):       
            for states, actions, rewards, dones, next_states in self.buffer.get_all().batch(batch_size):
                self.train_ppo(states, actions, rewards, dones, next_states)

                # self.rewards_metric(rewards)
            with self.train_summary_writer.as_default():
                tf.summary.scalar('actor_loss', self.actor_loss_metric.result(), step=epoch)
                tf.summary.scalar('critic_loss', self.critic_loss_metric.result(), step=epoch)
                tf.summary.scalar('entropy', self.entropy_metric.result(), step=epoch)
                tf.summary.scalar('advantages', self.advantages_metric.result(), step=epoch)
                tf.summary.scalar('returns', self.returns_metric.result(), step=epoch)
                # tf.summary.scalar('rewards', self.rewards_metric.result(), step=epoch)

        self.buffer.clean_buffer()

        self.actor_loss_metric.reset_states()
        self.critic_loss_metric.reset_states()
        self.entropy_metric.reset_states()
        self.advantages_metric.reset_states()
        self.returns_metric.reset_states()
        # self.rewards_metric.reset_states()
                
        # Copy new weights into old policy:
        self.actor_old_model.set_weights(self.actor_model.get_weights())
        self.critic_old_model.set_weights(self.critic_model.get_weights())

    def save_models(self,episode,identifier):
        time = utils.get_time_date()
        if os.path.exists(self.env_name+'/models') is False:
            os.mkdir(self.env_name+'/models')
        print('Saving models as -{}- {}'.format(identifier,time))
        self.actor_model.save(self.env_name+'/models/actor_model_'+str(episode)+identifier+time+'.h5')
        self.actor_old_model.save(self.env_name+'/models/actor_old_model_'+str(episode)+identifier+time+'.h5')
        self.critic_model.save(self.env_name+'/models/critic_model_'+str(episode)+identifier+time+'.h5')
        self.critic_old_model.save(self.env_name+'/models/critic_old_model_'+str(episode)+identifier+time+'.h5')

    def load_models(self,path):
        if os.path.exists(path+'actor_model.h5') is True and os.path.exists(path+'actor_model.h5') is True:
            print('Loading models . . . ')
            # self.actor_model.save(self.env_name+'/models/actor_model_'+str(episode)+identifier+time+'.h5')
            # self.actor_old_model.save(self.env_name+'/models/actor_old_model_'+str(episode)+identifier+time+'.h5')
            # self.critic_model.save(self.env_name+'/models/critic_model_'+str(episode)+identifier+time+'.h5')
            # self.critic_old_model.save(self.env_name+'/models/critic_old_model_'+str(episode)+identifier+time+'.h5')
        else:
            print('Models not found . . . ')
            exit()