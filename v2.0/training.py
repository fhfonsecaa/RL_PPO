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
from buffer import Buffer

if os.path.exists('plots') is False:
    os.mkdir('plots')

def plot(datas, marker, title, x_label, y_label, id):
    plt.plot(datas, marker)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.show(block=False)
    plt.savefig('plots/'+id+'_'+y_label+'.png')
    plt.pause(3)
    plt.close()
 
def run_episode(env, agent, state_dim, render, training_mode, updates_counter, updates_threshold):
    state = env.reset()     
    done = False
    total_reward = 0
    eps_time = 0

    while not done:
        action = [agent.get_action(state).numpy()]
        action_gym = np.array(action)

        observations, reward, done, _ = env.step(action_gym)
        
        eps_time += 1 
        updates_counter += 1
        total_reward += reward
          
        if training_mode:
            agent.save_eps(state, reward, action, done, observations) 
            
        state = observations     
                
        if render:
            env.render()
        
        if training_mode:
            if updates_counter % updates_threshold == 0:
                agent.update_ppo()
                updates_counter = 0
       
        if done:
            return total_reward, eps_time, updates_counter
    
def main():
    mujoco = False
    render = False
    save_models = False # If you want to save the agent, set this to True
    training_mode = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
    reward_threshold = None # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off
    update_threshold = 2000 # Number of iterations before update the Policy
    plot_batch_threshold = 1000 # How many episode you want to plot the result
    episode_max = 150000 # How many episodes to run

    if mujoco:
        env_name = "Humanoid-v2"
        epsilon_discount = 4.0e-5
    else:
        env_name = "MountainCarContinuous-v0"
        epsilon_discount = 4.0e-4

    env = gym.make(env_name)

    env.seed(69)
    np.random.seed(69)
    tf.random.set_seed(69)
    
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    epsilon = 0.9
    agent = Agent(state_dim, action_dim, env, epsilon, mujoco)  
    
    rewards = []   
    rewards_means = []   
    batch_rewards = []
    batch_solved_reward = []
    
    times = []
    batch_times = []

    updates_counter = 0
    
    for i_episode in range(1, episode_max + 1):
        try:
            total_reward, time, updates_counter = run_episode(env, agent, state_dim, render, training_mode, updates_counter, update_threshold)
            print('Episode {} Elapsed time: {} Total reward: {}  Epsilon: {}'.format(i_episode, time, int(total_reward), agent.get_epsilon()))
            batch_rewards.append(int(total_reward))
            batch_times.append(time)   
            epsilon -= epsilon_discount
            
            if epsilon >= 0.2 and training_mode:
                agent.set_epsilon(epsilon)
            if save_models:
                agent.save_models(i_episode,'')  
                                
            if reward_threshold:
                if len(batch_solved_reward) == 100:            
                    if np.mean(batch_solved_reward) >= reward_threshold :              
                        rewards = rewards + batch_rewards
                        times = times = batch_times                    

                        print('Task solved after {} episodes'.format(i_episode))
                        agent.save_models(i_episode,'solved')  
                        break
                    else:
                        del batch_solved_reward[0]
                        batch_solved_reward.append(total_reward)
                else:
                    batch_solved_reward.append(total_reward)
                
            if i_episode % plot_batch_threshold == 0:
                print('=====================')
                print('|-------Batch-------|')
                print('=====================')
                plot(batch_rewards,"+",'Rewards of batch until episode {}'.format(i_episode), 'Episodes','Rewards',str(i_episode)+'_Batch')
                plot(batch_times,".",'Times of batch until episode {}'.format(i_episode), 'Episodes','Times',str(i_episode)+'_Batch')
                
                rewards_mean = np.mean(batch_rewards)
                print('Max Reward:', np.max(batch_rewards))
                print('Min Reward:', np.min(batch_rewards))
                print('Avg Reward:', rewards_mean)
                print('')

                rewards = rewards + batch_rewards
                times = times = batch_times
                rewards_means.append(rewards_mean)
                    
                batch_rewards = []
                batch_times = []

                print('============================')
                print('|-------Accumulative-------|')
                print('============================')
                plot(rewards,"+",'Total rewards until episode {}'.format(i_episode), 'Episodes','Rewards',str(i_episode)+'_Total')
                plot(times,".",'Total times until episode {}'.format(i_episode), 'Episodes','Times',str(i_episode)+'_Total')
        except KeyboardInterrupt:
            print('Training loop interrupted, saving last models . . .')
            agent.save_models(i_episode,'forced') 
            plot(rewards_means,"ro-",'Average reward per batch until episode {}'.format(i_episode), 'Batchs','Rewards',str(i_episode)+'_BatchAverage')
                
            exit() 
    agent.save_models(episode_max,'finalized')
    plot(rewards_means,"ro-",'Average reward per batch until episode {}'.format(i_episode), 'Batchs','Rewards',str(i_episode)+'_BatchAverage')

if __name__ == '__main__':
    main()