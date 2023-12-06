import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
from collections import deque
import tensorflow as tf
import random

acoes = 6
interacoes = 25
arrayinteracoes = np.random.rand(interacoes)
print (arrayinteracoes)
QCenario=[]

F1 = np.array([.35, .3, .2, .1, .09])
F2 = np.array([.35*1.5, .3*1.5, .2*1.5, .1*.5, .09*.5])
F3 = np.array([.35*.5, .3*.5, .2*.5, .1*1.5, .09*1.5])


for acao in range(acoes ) :
  Q = np.array([0,0,0,0,0])
  Q = Q[np.newaxis,:]
  Q_ = Q.copy()

  for i in range(interacoes):
     if i <= 20:
       Q = Q + (1-Q)*F1*arrayinteracoes[acao]
     else:
       Q = Q + (1-Q)*F3*arrayinteracoes[acao]
     Q_ = np.concatenate([Q_,Q], axis=0)

  QCenario.append(Q_)

def create_model(imputDim, outPutDim, layers, lr):
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(
        layers[0], input_dim=imputDim ,activation="relu"))
    for i in range(1,len(layers)):
        model.add(tf.keras.layers.Dense(layers[i], activation="relu"))
    model.add(tf.keras.layers.Dense(outPutDim))
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=lr))
    model.summary()
    return model

def train ( episodes,DoneActions,DoneVal,epsilon,acoes,model,TypeReward,buffer_size,benginTrainDqn,batchDQN,epsilon_decay,gamma  ): 
 memory = deque(maxlen=buffer_size)
 target_model = tf.keras.models.clone_model(model)
 for i_episode in range(episodes):
     done = True 
     interacation = -1
     episode_return = 0
     action = 0
     
     while done: 
          interacation = interacation +1 
          state = np.array(QCenario[action][interacation])
          print(state)
          print("\r> DQN: Episode {}/{}, Step {}, Return {}".format(
                i_episode+1, episodes, action, episode_return), end="")
          if np.random.rand() <epsilon:
                print(acoes)
                action = random.randint(0,acoes-1)
          else:
                #if TypeReward == "MAX" : 
                 print('***** model')
                 print(model.predict(np.array([state])))
                 action = np.argmax(model.predict(np.array([state]))[0])
          
          state_new = QCenario[action][interacation]
          if TypeReward == "MAX" : 
             reward = np.sum(QCenario[acao][interacation+1]) 
           
          if interacation >= DoneActions -1 : 
              done = False
          if reward >=  4.10 : 
              done = False 
            
          episode_return = reward
          memory.append((state,action,reward,state_new,done))
             
          if len(memory) > benginTrainDqn:
                experience_sample = random.sample(memory, batchDQN)
                
                x = np.array([e[0] for e in experience_sample])

                # Construct target
                y = model.predict(x)
                print (' ** y')
                print(y)
                x2 = np.array([e[3] for e in experience_sample])
                Q2 = gamma*np.max(target_model.predict(x2), axis=1)
               
                for i,(s,a,r,s2,d) in enumerate(experience_sample):
                    print (' ** a')
                    print(a)
                    print(i)
                    y[i][a] = r
                    if not d:
                        y[i][a] += Q2[i]

                # Update
                model.fit(x, y, batch_size=batchDQN, epochs=1, verbose=0)

               
                target_model.set_weights(model.get_weights())

          state = state_new
          

           # Epsilon decay
          epsilon *= epsilon_decay
        
      
layers =  [32,32]
actions = 5 
imputSize = 5 
episodes = 21
DoneActions = 25 
DoneVal = 4.50 
TypeReward = "MAX"
epsilon = 1
buffer_size = 500 
benginTrainDqn = 6
batchDQN = 6
epsilon_decay = 0.95
gamma = 0.9
model = create_model(imputSize, actions, layers, 1e-3)
  
train ( episodes,DoneActions,DoneVal,epsilon,actions,model,TypeReward,buffer_size,benginTrainDqn,batchDQN,epsilon_decay,gamma  )

  