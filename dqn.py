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


env = gym.make('MountainCar-v0')

input_shape = env.observation_space.shape[0]
num_actions = env.action_space.n

value_network = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(input_shape,)),
    tf.keras.layers.Dense(32, activation='relu'),tf.keras.layers.Dense(32, activation='linear'),
    tf.keras.layers.Dense(num_actions)
])

# Set up the optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
#value_network = tf.keras.models.load_model('keras')

num_episodes = 50
epsilon = 1
gamma = 0.9
state = env.reset()
batch = 50
replay = deque(maxlen=2000)
epoch = 0
alpha = 0.1

for episode in range(num_episodes):
    state = env.reset()

    # Run the episode
    while True:
      
        value_function = value_network.predict(np.array([state]),verbose=0)[0]
        
        
        if np.random.rand()>epsilon:
            action = np.argmax(value_function)
            
        else:
            action = np.random.choice(num_actions)


        next_state, reward, done, _ = env.step(action)
        
        done = 1 if done else 0
      
        replay.append((state,action,reward,next_state,done))

        state = next_state


        if done:        
            break
    
        if len(replay)>batch:
            with tf.GradientTape() as tape:
                batch_ = random.sample(replay,batch)
                q_value1 = value_network(tf.convert_to_tensor([x[0] for x in batch_]))
                q_value2 = value_network(tf.convert_to_tensor([x[3] for x in batch_]))
                
                reward = tf.convert_to_tensor([x[2] for x in batch_])
                action = tf.convert_to_tensor([x[1] for x in batch_])
                done =   tf.convert_to_tensor([x[4] for x in batch_])
      
                
                actual_q_value1 = tf.cast(reward,tf.float64) + tf.cast(tf.constant(alpha),tf.float64)*(tf.cast(tf.constant(gamma),tf.float64)*tf.cast((tf.constant(1)-done),tf.float64)*tf.cast(tf.reduce_max(q_value2),tf.float64))           
                loss = tf.cast(tf.gather(q_value1,action,axis=1,batch_dims=1),tf.float64)
                loss = loss - actual_q_value1
                loss = tf.reduce_mean(tf.math.pow(loss,2))

        
                grads = tape.gradient(loss, value_network.trainable_variables)
                optimizer.apply_gradients(zip(grads, value_network.trainable_variables))

                print('Epoch {} done with loss {} !!!!!!'.format(epoch,loss))

                value_network.save('keras/')
                if epoch%100==0:
                    epsilon*=0.999
                epoch+=1


#
#pygame essentials
pygame.init()
DISPLAYSURF = pygame.display.set_mode((625,400),0,32)
clock = pygame.time.Clock()
pygame.display.flip()

#openai gym env
env = gym.make('MountainCar-v0')
input_shape = env.observation_space.shape[0]
num_actions = env.action_space.n
state = env.reset()

done = False
count=0
done=False
steps = 0
#loading trained model
value_network = tf.keras.models.load_model('keras')

def print_summary(text,cood,size):
        font = pygame.font.Font(pygame.font.get_default_font(), size)
        text_surface = font.render(text, True, (125,125,125))
        DISPLAYSURF.blit(text_surface,cood)

while count<10000 :
    pygame.event.get()
    steps+=1
    
    
    for event in pygame.event.get():
                if event.type==QUIT:
                                pygame.quit()
                                raise Exception('training ended')
    # Get the action probabilities from the policy network
    action = np.argmax(value_network.predict(np.array([state]))[0])


    obs, reward, done, info = env.step(action) # take a step in the environment
    image = env.render(mode='rgb_array') # render the environment to the screen
   
    #convert image to pygame surface object
    image = Image.fromarray(image,'RGB')
    mode,size,data = image.mode,image.size,image.tobytes()
    image = pygame.image.fromstring(data, size, mode)

    DISPLAYSURF.blit(image,(0,0))
    print_summary('Step {}'.format(steps),(10,10),15)
    pygame.display.update()
    clock.tick(100)
    count+=1
    if done:
        print_summary('Episode ended !!!'.format(steps),(100,100),30)
        
        state = env.reset()
        done = False 
    state = next_state

pygame.quit()    