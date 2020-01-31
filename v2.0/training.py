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

from agent import PPOAgent
from buffer import Buffer
import pathlib

def check_folder(env_name):
    pathlib.Path(env_name+'/plots').mkdir(parents=True, exist_ok=True)

def plot(env_name, datas, marker, title, x_label, y_label, id):
    plt.plot(datas, marker)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.show(block=False)
    plt.savefig(env_name+'/plots/'+id+'_'+y_label+'.png')
    plt.pause(1)
    plt.close()
 
def run_episode(env, agent, state_dim, render, training_mode, updates_counter, updates_threshold):
    state = env.reset()     
    done = False
    total_reward = 0
    eps_time = 0

    while not done:
        if render:
            env.render()

        action = [agent.get_action(state).numpy()]
        action_gym = np.array(action)

        observations, reward, done, _ = env.step(action_gym)
        
        eps_time += 1 
        updates_counter += 1
        total_reward += reward
          
        if training_mode:
            agent.save_eps(state, reward, action, done, observations) 
            if updates_counter % updates_threshold == 0:
                agent.update_ppo()
                updates_counter = 0
        state = observations     
       
        if done:
            return total_reward, eps_time, updates_counter
    
def main():
    mujoco = True
    render = False
    save_models = False # Save the models 
    training_mode = True # Train the agent or test a memory model
    reward_threshold = None 
    # reward_threshold = 290 

    update_threshold = 800 # Iterations before update the Policy
    plot_batch_threshold = 500 # Espisodes included in the partial plot
    episode_max = 30000 

    # update_threshold = 1000 # Iterations before update the Policy
    # plot_batch_threshold = 100 # Episodes included in the partial plot
    # episode_max = 3000 

    if mujoco:
        env_name = 'Humanoid-v2'
        epsilon_discount = 5.0e-3
    else:
        env_name = 'MountainCarContinuous-v0'
        epsilon_discount = 4.0e-4

    env = gym.make(env_name)
    check_folder(env_name)
    env.seed(69)
    np.random.seed(69)
    tf.random.set_seed(69)
    
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    epsilon = 0.9
    agent = PPOAgent(state_dim, action_dim, env, epsilon, mujoco)  

    if not training_mode:
        path = ''
        agent.load_models(path)
    
    rewards = []   
    rewards_means = []   
    batch_rewards = []
    batch_solved_reward = []
    
    times = []
    batch_times = []

    updates_counter = 0
    
    tb_writer = agent.get_summary_writer()
    rewards_metric = tf.keras.metrics.Mean(name='rewards_metric')

    for epis in range(1, episode_max + 1):
        try:
            total_reward, time, updates_counter = run_episode(env, agent, state_dim, render, training_mode, updates_counter, update_threshold)
            print('Episode {} Elapsed time: {} Total reward: {}  Epsilon: {}'.format(epis, time, int(total_reward), agent.get_epsilon()))
            batch_rewards.append(int(total_reward))
            batch_times.append(time)   
            epsilon -= epsilon_discount

            rewards_metric(total_reward)
            with tb_writer.as_default():
                tf.summary.scalar('rewards', rewards_metric.result(), step=epis)
            rewards_metric.reset_states()
            
            if epsilon >= 0.2 and training_mode:
                agent.set_epsilon(epsilon)
            if save_models:
                agent.save_models(epis,'')  
                
            if epis % plot_batch_threshold == 0:
                print('=====================')
                print('|-------Batch-------|')
                print('=====================')
                plot(env_name,batch_rewards,"+",'Rewards of batch until episode {}'.format(epis), 'Episodes','Rewards',str(epis)+'_Batch')
                plot(env_name,batch_times,".",'Times of batch until episode {}'.format(epis), 'Episodes','Times',str(epis)+'_Batch')
                
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
                plot(env_name,rewards,"+",'Total rewards until episode {}'.format(epis), 'Episodes','Rewards',str(epis)+'_Total')
                plot(env_name,times,".",'Total times until episode {}'.format(epis), 'Episodes','Times',str(epis)+'_Total')

            if reward_threshold:
                if len(batch_solved_reward) == 100:            
                    if np.mean(batch_solved_reward) >= reward_threshold :              
                        rewards = rewards + batch_rewards
                        times = times = batch_times                    

                        print('============================')
                        print('Reward threshold reached after {} episodes'.format(epis))
                        print('============================')
                        agent.save_models(epis,'solved')  
                        break
                    else:
                        del batch_solved_reward[0]
                        batch_solved_reward.append(total_reward)
                else:
                    batch_solved_reward.append(total_reward)
        
        except KeyboardInterrupt:
            print('Training loop interrupted, saving last models . . .')
            agent.save_models(epis,'forced') 
            plot(env_name,rewards_means,"ro-",'Average reward per batch until episode {}'.format(epis), 'Batchs','Rewards',str(epis)+'_BatchAverage')
                
            exit() 
    agent.save_models(episode_max,'finalized')
    plot(env_name,rewards_means,"ro-",'Average reward per batch until episode {}'.format(epis), 'Batchs','Rewards',str(epis)+'_BatchAverage')

if __name__ == '__main__':
    main()