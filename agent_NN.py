import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from collections import deque

import tensorflow as tf
#physical_devices = tf.config.list_physical_devices('GPU')  # for Ian's GPU 
#tf.config.experimental.set_memory_growth(physical_devices[0], True) # for Ian's GPU 

class DQN():
    """ A DNN """
    def __init__(self, state_size, action_size):
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.memory  = deque(maxlen=2000)

        self.learning_rate = 0.005
        self.tau = .125

        self.model        = self.create_model()
        self.target_model = self.create_model()
    
    
    def create_model(self):
        model   = Sequential()
        #model.add(Dense(12, input_dim=self.state_size, activation="relu"))
        #model.add(Dense(24, activation="relu"))
        #model.add(Dense(self.action_size))

        #model.add(InputLayer(batch_input_shape=(1, self.state_size)))
        #model.add(Dense(10, activation='sigmoid'))
        #model.add(Dense(self.action_size, activation='linear'))
        #model.compile(loss='mse', optimizer='adam', metrics=['mae'])


        model.add(Dense(output_dim = self.state_size, input_dim=self.state_size, activation="relu"))
        model.add(Dense(output_dim = self.state_size, activation="relu"))
        model.add(Dense(self.action_size))

        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate))
        return model


    
    def sgd(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

        #mse = np.mean((Yhat - Y) ** 2)
        #self.losses.append(mse)
        
    
        
#     def load_weights(self, filepath):
#         npz = np.load(filepath)
#         self.W = npz["W"]
#         self.b = npz["b"]

#     def save_weights(self, filepath):
#         np.savez(filepath, W=self.W, b=self.b)


class DQNAgent(object):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        self.memory  = deque(maxlen=2000)

        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.dqn = DQN(state_size, action_size)

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_size)
        return np.argmax(self.dqn.model.predict(state)[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def train(self, state, action, reward, next_state, done):
        
        self.remember(state, action, reward, next_state, done)

        batch_size = 3
        if len(self.memory) < batch_size: 
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, next_state, done = sample
            target = self.dqn.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.dqn.target_model.predict(next_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.dqn.model.fit(state, target, epochs=1, verbose=0)

        self.dqn.sgd()

    
#     def train(self, state, action, reward, next_state, done):
#         if done:
#             target = reward
#         else:
#             target = reward + self.gamma * np.amax(
#                 self.model.predict(next_state), axis=1
#             )

#         target_full = self.model.predict(state)
#         target_full[0, action] = target

#         # Run one training step
#         self.model.sgd(state, target_full)

#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.dqn.model.load_weights(name)

    def save(self, name):
        self.dqn.model.save_weights(name)
