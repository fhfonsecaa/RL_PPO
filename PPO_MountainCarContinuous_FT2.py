#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 10:57:16 2019

@author: fhfonsecaa
"""
import gym

import scipy.stats

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2

import tensorflow.keras.layers as kl
import numpy as np
print(tf.__version__)

gamma = 0.99
lambda_ = 0.95
clipping_val = 0.2
critic_discount = 0.5
entropy_beta = 0.001

def test_reward():
    state = env.reset()
    done = False
    total_reward = 0 
    print('Testing . . . ')
    limit = 0
    while not done:
        state_imput = tf.expand_dims(state, 0)
        # actions_prob  #TODO dist
        action = model_actor.predict([state_imput, dummy_n, dummy_1, dummy_1, dummy_1], steps = 1).reshape(1,)#TODO dist for discrete model
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        limit += 1
        if limit > 20:
            break
    return total_reward

        
def get_advantages(values, masks, rewards):
    print('Calculating Advantages')
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma*values[i+1]*masks[i] - values[i]
        gae = delta + gamma*lambda_*masks[i]*gae
        returns.insert(0, gae+values[i])
    adv = np.array(returns - values[i-1])
    return returns, (adv - np.mean(adv))/(np.std(adv)+1e-10)

def ppo_loss(oldpolicy_probs, advantages, rewards, values):
    def loss(y_true, y_pred):
        newpolicy_probs = y_pred

        loss_adv = oldpolicy_probs[0,0]

        loss_adv = tf.print(loss_adv, [loss_adv], 'loss_adv: ')

        # newpolicy_means = newpolicy_probs[0,0]
        # newpolicy_sigmas = abs(newpolicy_probs[0,1])
        oldpolicy_means = oldpolicy_probs[0,0]
        # oldpolicy_sigmas = oldpolicy_probs[0,1]

        # actions_min = -2
        # actions_max = 2
        # actions_range = np.linspace(actions_min, actions_max, 100)
        # newpolicy_probs = scipy.stats.norm.pdf(actions_range,newpolicy_means,newpolicy_sigmas)
        oldpolicy_probs = scipy.stats.norm.pdf(actions_range,oldpolicy_means,oldpolicy_sigmas)



        ratio = tf.math.exp(tf.math.log(newpolicy_probs+1e-10)-tf.math.log(oldpolicy_probs+1e-10))
        p1 = ratio*advantages
        p2 = tf.clip_by_value(ratio, clip_value_min=1-clipping_val, clip_value_max=1+clipping_val)*advantages
        actor_loss = tf.keras.backend.mean(tf.keras.backend.minimum(p1,p2))
        critic_loss = tf.keras.backend.mean(tf.keras.backend.square(rewards - values))
        total_loss = critic_discount*critic_loss + actor_loss - entropy_beta*tf.keras.backend.mean(
            -(newpolicy_probs*tf.math.log(newpolicy_probs + 1e-10))
        )
        return total_loss
    return loss

# def ppo_loss_ind(oldpolicy_probs, advantages, rewards, values):
#     # newpolicy_probs = y_pred
#     dummy_n_l = tf.reshape(np.zeros((ppo_steps,n_actions)), [-1, 2])
#     dummy_1_l = np.zeros((ppo_steps,1))
#     newpolicy_probs = model_actor.predict([states, dummy_n_l, dummy_1_l, dummy_1_l, dummy_1_l])
#     print('newpolicy_probs Norm')
#     print(newpolicy_probs)
#     print('oldpolicy_probs Norm')
#     print(oldpolicy_probs)
#     print('---------------------')
#     newpolicy_means = newpolicy_probs[0,0]
#     newpolicy_sigmas = abs(newpolicy_probs[0,1])
#     oldpolicy_means = oldpolicy_probs[0,0]
#     oldpolicy_sigmas = abs(oldpolicy_probs[0,1])

#     print(newpolicy_means)
#     print(newpolicy_sigmas)
#     print(oldpolicy_means)
#     print(oldpolicy_sigmas)
#     print('---------------------')
#     print('---------------------')
#     print('---------------------')
#     actions_min = -2
#     actions_max = 2
#     actions_range = np.linspace(actions_min, actions_max, 100)
#     newpolicy_probs = scipy.stats.norm.pdf(actions_range,newpolicy_means,newpolicy_sigmas)
#     print('newpolicy_probs')
#     print(newpolicy_probs)
#     oldpolicy_probs = scipy.stats.norm.pdf(actions_range,oldpolicy_means,oldpolicy_sigmas)
#     print('oldpolicy_probs')
#     print(oldpolicy_probs)
#     print('---------------------')
#     print('---------------------')
#     print('LOG newpolicy_probs')
#     print(tf.math.log(newpolicy_probs+1e-10))
#     print('LOG oldpolicy_probs')
#     print(tf.math.log(oldpolicy_probs+1e-10))

#     ratio = tf.math.exp(tf.math.log(newpolicy_probs+1e-10)-tf.math.log(oldpolicy_probs+1e-10))
#     print(ratio)
#     print(advantages)
#     p1 = ratio*advantages
#     print(p1)
#     p2 = tf.clip_by_value(ratio, clip_value_min=1-clipping_val, clip_value_max=1+clipping_val)*advantages
#     actor_loss = tf.keras.backend.mean(tf.keras.backend.minimum(p1,p2))
#     critic_loss = tf.keras.backend.mean(tf.keras.backend.square(rewards - values))
#     total_loss = critic_discount*critic_loss + actor_loss - entropy_beta*tf.keras.backend.mean(
#         -(newpolicy_probs*tf.math.log(newpolicy_probs + 1e-10))
#     )
#     print(total_loss)
#     return total_loss

def get_model_actor(input_dims, output_dims):
    state_input = kl.Input(shape=input_dims)

    oldpolicy_probs = kl.Input(shape=(output_dims,))
    advantages = kl.Input(shape=(1,))
    rewards = kl.Input(shape=(1,))
    values = kl.Input(shape=(1,))

    x = kl.Dense(512, kernel_initializer='random_uniform', activation='tanh', name='fc0')(state_input)
    x = kl.Dense(256, kernel_initializer='random_uniform', activation='tanh', name='fc1')(x)
    out_actions = kl.Dense(output_dims, activation='linear', name='predictions')(x)

    model = Model(inputs=[state_input, oldpolicy_probs, advantages, rewards, values], outputs=[out_actions])
    # model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    model.compile(optimizer=Adam(lr=1e-4), loss=[ppo_loss(oldpolicy_probs=oldpolicy_probs, advantages=advantages,
                                                          rewards=rewards, values=values)])
    return model

def get_model_critic(input_dims):
    state_input = kl.Input(shape=input_dims)

    x = kl.Dense(512, kernel_initializer='random_uniform', activation='tanh', name='fc0')(state_input)
    x = kl.Dense(256, kernel_initializer='random_uniform', activation='tanh', name='fc1')(x)
    out_actions = kl.Dense(1, activation='tanh', name='predictions')(x)

    model = Model(inputs=[state_input], outputs=[out_actions])
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    return model

def sample_action(env, mu, sigma):
    noise = np.random.normal(0,1,1)
    action = np.random.normal(mu, sigma) + noise
    print(action)
    action = np.clip(action, env.action_space.low[0], env.action_space.high[0])
    print(action)
    return np.array(action).reshape(1,)

if __name__ == '__main__':
    x_min = -2
    x_max = 2

    mean = 0.8
    std = 1

    x = np.linspace(x_min, x_max, 100)
    y = scipy.stats.norm.pdf(x,mean,std)

    mean = -0.2
    std = 1e-2

    z = scipy.stats.norm.pdf(x,mean,std)
    print(z)

    k = tf.math.log(z)
    print(k)

    # plt.plot(x,y, color='coral')
    # plt.plot(x,z, color='red')
    # plt.plot(x,k, color='blue')

    # plt.grid()

    # plt.xlim(x_min,x_max)
    # plt.ylim(-10,1)

    # plt.xlabel('x')
    # plt.ylabel('Normal Distribution')
    # plt.show()

    # input('holis')

    ppo_steps = 1

    env = gym.make('MountainCarContinuous-v0')
    state = env.reset()
    state_dims = env.observation_space.shape
    n_actions = 2 #For Car Box Continuous Mu Sigma

    model_actor = get_model_actor(input_dims=state_dims, output_dims=n_actions)
    model_critic = get_model_critic(input_dims=state_dims)
    model_actor.summary()

    dummy_n = tf.reshape(np.zeros((n_actions)), [-1, 2])
    dummy_1 = np.zeros((1))

    best_reward = 0
    target_reached = False
    iters = 0
    max_iters = 1000

    while not target_reached and iters < max_iters:

        state = env.reset()
        states = []
        actions = []
        actions_deter = []
        actions_probs = []
        values = []
        masks = []
        rewards = []

        state_imput = None

        for itr in range(ppo_steps):
            observation = env.render()
            action = env.action_space.sample()

            state_imput = tf.reshape(state, [-1, 2])

            action_dist = model_actor.predict([state_imput, dummy_n, dummy_1, dummy_1, dummy_1], steps = 1)
            mu = action_dist[0][0]
            sigma = abs(action_dist[0][1])
            # action_dist[0][1] = abs(action_dist[0][1])

            action = sample_action(env, mu, sigma)
            q_value = model_critic.predict(state_imput, steps = 1)

            observation, reward, done, info = env.step(action)
            print('itr: ' + str(itr) + ', action=' + str(action) + ', reward=' + str(reward) + ', q val=' + str(q_value))
            mask = not done

            action_deter = np.zeros(n_actions)
            action_deter[0] = mu

            states.append(state)
            actions.append(action)
            actions_deter.append(action_deter)
            values.append(q_value)
            masks.append(mask)
            rewards.append(reward)
            actions_probs.append(action_dist)

            state = observation
            if done:
                env.reset()

        q_value = model_critic.predict(state_imput, steps = 1)
        values.append(q_value)

        returns, advantages = get_advantages(values, masks, rewards)



        # print('advantages')
        # print(advantages.reshape(-1,1).shape)
        # print(advantages.reshape(-1,1))

        # print('rewards')
        # print(len(rewards))
        # print(rewards)

        # print('masks')
        # print(len(masks))
        # print(masks)

        # print('values')
        # print(len(np.reshape(values[:-1], newshape = (-1, 1))))
        # print(np.reshape(values[:-1], newshape = (-1, 1)))
        # print(values[:-1])
        # print(n_actions)

        # input('Waiting for check')
        print('Fitting Models')
        # ppo_loss_ind(tf.reshape(actions_probs, [-1, 2]), advantages.reshape(-1,1), rewards, np.reshape(values[:-1], newshape = (-1, 1)))
        # input('Waiting for check')

        input('Waiting for check')

        model_actor.fit(
            [states, tf.reshape(actions_probs, [-1, 2]), advantages.reshape(-1,1), rewards, np.reshape(values[:-1], newshape = (-1, 1))],
            [tf.reshape(actions_deter, [-1, 2])],verbose=True, shuffle=True, epochs=8)
        model_critic.fit(
            [states], 
            [np.reshape(returns, newshape = (-1, 1))],verbose=True, shuffle=True, epochs=8)
        input('Waiting for check')

        # avg_reward = np.mean([test_reward() for _ in range(5) ])
        # print('Total test reward {}'.format(avg_reward))
        # if avg_reward > best_reward:
        #     print('Best reward {}'.format(best_reward))

    env.close()