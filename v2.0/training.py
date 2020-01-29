#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 10:49:55 2019

@author: fhfonsecaa
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import gym
import mujoco_py

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import tensorflow_probability as tfp

from agent import Agent
from utility import ProbUtils
from buffer import Buffer

if os.path.exists('plots') is False:
    os.mkdir('plots')

def plot(datas, marker, title, y_label, id):
    plt.plot(datas, marker)
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel(y_label)
    plt.grid()
    plt.show(block=False)
    plt.savefig('plots/'+id+'_'+y_label+'.png')
    plt.pause(3)
    plt.close()
    
    print('Max '+y_label+':', np.max(datas))
    print('Min '+y_label+':', np.min(datas))
    print('Avg '+y_label+':', np.mean(datas))

def run_episode(env, agent, state_dim, render, training_mode, t_updates, n_update):
    state = env.reset()     
    done = False
    total_reward = 0
    eps_time = 0

    while not done:
        action = [agent.get_action(state).numpy()]
        action_gym = 2 * np.array(action)

        if np.random.uniform(0,1) > agent.epsilon:        
            observations, reward, done, _ = env.step(action_gym)
        else:
            observations, reward, done, _ = env.step(env.action_space.sample())
        
        eps_time += 1 
        t_updates += 1
        total_reward += reward
          
        if training_mode:
            agent.save_eps(state, reward, action, done, observations) 
            
        state = observations     
                
        if render:
            env.render()
        
        if training_mode:
            if t_updates % n_update == 0:
                agent.update_ppo()
                t_updates = 0
       
        if done:
            return total_reward, eps_time, t_updates

def run_episode_humanoid(env, agent, state_dim, render, training_mode, t_updates, n_update):
    state = env.reset()     
    done = False
    total_reward = 0
    eps_time = 0
    n = 0

    while n < 100:
        action = [agent.act(state).numpy()]
        action_gym = 2 * np.array(action)

        if np.random.uniform(0,1) > agent.epsilon:        
            next_state, reward, done, _ = env.step(action_gym)
        else:
            next_state, reward, done, _ = env.step(env.action_space.sample())
        
        eps_time += 1 
        t_updates += 1
        total_reward += reward
          
        if training_mode:
            agent.save_eps(state, reward, action, done, next_state) 
            
        state = next_state     
                
        if render:
            env.render()
        
        if training_mode:
            if t_updates % n_update == 0:
                agent.update_ppo()
                t_updates = 0
        n += 1

    return total_reward, eps_time, t_updates
    
def main():

    # load_weights = True # If you want to load the agent, set this to True
    save_weights = False # If you want to save the agent, set this to True
    training_mode = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
    reward_threshold = None # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off
    mujoco = True
    render = True # If you want to display the image.
    update_threshold = 2048 # How many episode before you update the Policy
    plot_batch_threshold = 100 # How many episode you want to plot the result
    episode_max = 10000 # How many episode you want to run

    if mujoco:
        env_name = "Humanoid-v2"
    else:
        env_name = "MountainCarContinuous-v0"

    env = gym.make(env_name)

    env.seed(69)
    np.random.seed(69)
    tf.random.set_seed(69)
    
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    epsilon = 0.8
    agent = Agent(state_dim, action_dim, env, epsilon, mujoco)  
    
    rewards = []   
    batch_rewards = []
    batch_solved_reward = []
    
    times = []
    batch_times = []

    updates_counter = 0
    
    for i_episode in range(1, episode_max + 1):
        total_reward, time, updates_counter = run_episode(env, agent, state_dim, render, training_mode, updates_counter, update_threshold)
        print('Episode {} \t t_reward: {} \t time: {} \t epsilon: {} \t'.format(i_episode, int(total_reward), time, agent.get_epsilon()))
        batch_rewards.append(int(total_reward))
        batch_times.append(time)   
        epsilon -= 4.0e-4
        if epsilon >= 0.2:
            agent.set_epsilon(epsilon)
        if save_weights:
            agent.save_weights()  
                            
        if reward_threshold:
            if len(batch_solved_reward) == 100:            
                if np.mean(batch_solved_reward) >= reward_threshold :              
                    for reward in batch_rewards:
                        rewards.append(reward)

                    for time in batch_times:
                        times.append(time)                    

                    print('You solved task after {} episode'.format(len(rewards)))
                    break
                else:
                    del batch_solved_reward[0]
                    batch_solved_reward.append(total_reward)
            else:
                batch_solved_reward.append(total_reward)
            
        if i_episode % plot_batch_threshold == 0 and i_episode != 0:
            print('Batch')
            plot(batch_rewards,"+",'Rewards of batch until episode {}'.format(i_episode),'Rewards',str(i_episode)+'_Batch')
            plot(batch_times,".",'Times of batch until episode {}'.format(i_episode),'Times',str(i_episode)+'_Batch')
            
            rewards = rewards + batch_rewards
            times = times = batch_times
                
            batch_rewards = []
            batch_times = []

            print('Accumulative')
            plot(rewards,"+",'Total rewards until episode {}'.format(i_episode),'Rewards',str(i_episode)+'_Total')
            plot(times,".",'Total times until episode {}'.format(i_episode),'Times',str(i_episode)+'_Total')
            
if __name__ == '__main__':
    main()