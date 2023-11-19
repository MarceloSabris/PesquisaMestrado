import tensorflow as tf
import numpy as np
import gym
import math
from PIL import Image
import pygame, sys
from pygame.locals import *
from tensorflow import keras
from collections import deque
import random



class DQNAgent:
    def __init__(self, input_shape, num_actions):
        self.input_shape = input_shape
        self.num_actions = num_actions
        
 
    def _build_model(self):
        model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(self.input_shape,)),
        tf.keras.layers.Dense(32, activation='relu'),tf.keras.layers.Dense(32, activation='linear'),
        tf.keras.layers.Dense(self.num_actions)
        ])
        return model
 
    def remember(self, state, action, reward, next_state, done): 
        self.memory.append((state, action,
                            reward, next_state, done))

    def train(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward # if done 
            if not done:
                target = (reward +
                          self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0) 
        if self.epsilon & self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.rand() == self.epsilon:
                return random.randrange(self.action_size) 
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def save(self, name): 
        self.model.save_weights(name)

for e in range(n_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])

agent = DQNAgent(state_size, action_size)
done = False 
time = 0
while not done:
    #env.render()
    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)
    reward = reward if not done else -10
    next_state = np.reshape(next_state, [1, state_size]) 
    agent.remember(state, action, reward, next_state, done)
    state = next_state
    if done:
        print("episode: {}/{}, score: {}, e: {:.2}"
              .format(e, n_episodes-1, time, agent.epsilon))
    time += 1
if len(agent.memory) > batch_size:
    agent.train(batch_size) 
if e % 50 == 0:
    agent.save(output_dir + "weights_"
               + "{:04d}".format(e) + ".hdf5")