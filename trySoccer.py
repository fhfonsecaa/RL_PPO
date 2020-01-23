#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 10:57:16 2019

@author: fhfonsecaa
"""
import gym

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

def ppo_continuous_loss(oldpolicy_probs, advantages, rewards, values):
    def loss(y_true, y_pred):
        prob = tf.keras.backend.sum(y_true * y_pred)
        old_prob = tf.keras.backend.sum(y_true * oldpolicy_probs)
        r = prob/(old_prob + 1e-10)
        mean = tf.keras.backend.mean(tf.keras.backend.minimum(r * advantages, tf.clip_by_value(r, clip_value_min=0.8, clip_value_max=1.2) * advantages) + entropy_beta * -(prob * tf.math.log(prob + 1e-10)))
        loss = -tf.math.log(prob + 1e-10) * mean
        print(mean)
        print(loss)
        return loss
        # return -tf.math.log(prob + 1e-10) * tf.keras.backend.mean(tf.keras.backend.minimum(r * advantages, tf.clip_by_value(r, clip_value_min=0.8, clip_value_max=1.2) * advantages) + entropy_beta * -(prob * tf.math.log(prob + 1e-10)))
    return loss

def get_model_actor(input_dims, output_dims):
    state_input = kl.Input(shape=input_dims) #(2,)

    oldpolicy_probs = kl.Input(shape=(output_dims,))
    advantages = kl.Input(shape=(1,))
    rewards = kl.Input(shape=(1,))
    values = kl.Input(shape=(1,))

    x = kl.Dense(512, kernel_initializer='random_uniform', activation='tanh', name='fc0')(state_input)
    x = kl.Dense(256, kernel_initializer='random_uniform', activation='tanh', name='fc1')(x)
    out_actions = kl.Dense(n_actions, activation='linear', name='predictions')(x)

    model = Model(inputs=[state_input, oldpolicy_probs, advantages, rewards, values], outputs=[out_actions])
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    # model.compile(optimizer=Adam(lr=1e-4), loss=[ppo_loss(oldpolicy_probs=oldpolicy_probs, advantages=advantages,
                                                        #   rewards=rewards, values=values)])
    return model

def get_model_critic(input_dims):
    state_input = kl.Input(shape=input_dims)

    x = kl.Dense(512, kernel_initializer='random_uniform', activation='tanh', name='fc0')(state_input)
    x = kl.Dense(256, kernel_initializer='random_uniform', activation='tanh', name='fc1')(x)
    out_actions = kl.Dense(1, activation='tanh', name='predictions')(x)

    model = Model(inputs=[state_input], outputs=[out_actions])
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    return model

if __name__ == '__main__':
    ppo_steps = 15


    env = gym.make('MountainCarContinuous-v0')
    state = env.reset()
    print('state')
    print(state)
    state_dims = env.observation_space.shape
    n_actions = int(env.action_space.shape[0]) #For Car Box
    # n_actions = env.action_space

    model_actor = get_model_actor(input_dims=state_dims, output_dims=n_actions)
    model_critic = get_model_critic(input_dims=state_dims)

    print(state_dims)

    dummy_n = np.zeros((n_actions))
    dummy_1 = np.zeros((1))

    best_reward = 0
    target_reached = False
    iters = 0
    max_iters = 1000

    while not target_reached and iters < max_iters:

        state = env.reset()
        states = []
        actions = []
        # actions_onehot = []
        # actions_probs = []
        values = []
        masks = []
        rewards = []

        state_imput = None

        for itr in range(ppo_steps):
            observation = env.render()
            print('Observation')
            print(observation)
            noise = np.random.normal(0,2,1)
            print('Noise')
            print(noise)
            # action = 100
            action = env.action_space.sample() + noise

            print(action)

            state_imput = tf.expand_dims(state, 0)
            action = model_actor.predict([state_imput, dummy_n, dummy_1, dummy_1, dummy_1], steps = 1).reshape(1,) + noise
            q_value = model_critic.predict(state_imput, steps = 1)

            print('')
            print('Action')
            print(action)
            print('q_value')
            print(q_value)

            observation, reward, done, info = env.step(action)
            print('datos obtenidos del env')
            print(observation, reward, done, info )
            mask = not done

            states.append(state)
            actions.append(action)
            # actions_onehot.append(action_onehot)
            values.append(q_value)
            masks.append(mask)
            rewards.append(reward)
            # actions_probs.append(action_dist)

            state = observation
            if done:
                env.reset()

        q_value = model_critic.predict(state_imput, steps = 1)
        values.append(q_value)

        returns, advantages = get_advantages(values, masks, rewards)

        # print(len(states))
        # print(len(actions))
        # print(advantages.shape)
        # print(len(rewards))
        # print(len(values[:-1]))
        # print(n_actions)
        # print('-------------------')
        # print('states')
        # print(len(states))
        # print(states)

        print('actions')
        print(len(actions))
        print(actions)

        print('advantages')
        print(advantages.reshape(-1,1).shape)
        print(advantages.reshape(-1,1))

        print('rewards')
        print(len(rewards))
        print(rewards)

        print('masks')
        print(len(masks))
        print(masks)

        print('values')
        print(len(np.reshape(values[:-1], newshape = (-1, n_actions))))
        print(np.reshape(values[:-1], newshape = (-1, n_actions)))
        print(n_actions)

        input('Waiting for check')
        print('Fitting Models')
        model_actor.fit(
            [states, actions, advantages.reshape(-1,1), rewards, np.reshape(values[:-1], newshape = (-1, n_actions))],
            [np.reshape(actions, newshape = (-1, n_actions))],verbose=True, shuffle=True, epochs=8)
        model_critic.fit(
            [states], 
            [np.reshape(returns, newshape = (-1, 1))],verbose=True, shuffle=True, epochs=8)
        input('Waiting for check')

        # avg_reward = np.mean([test_reward() for _ in range(5) ])
        # print('Total test reward {}'.format(avg_reward))
        # if avg_reward > best_reward:
        #     print('Best reward {}'.format(best_reward))

    env.close()

    