from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from GlobalVariables import GlobalVariables
from keras.layers import Dense, Conv2D, Flatten,Conv1D, MaxPooling2D,Convolution2D,GlobalAveragePooling2D
from keras import optimizers
import random
import numpy as np
from keras.layers import MaxPooling1D,GlobalAveragePooling1D,Dropout,LSTM,TimeDistributed,AveragePooling1D,Embedding,Activation

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

parameter=GlobalVariables
grid_size=GlobalVariables
options=GlobalVariables

class DQNAgent:
    def __init__(self,env):
        if(options.use_samples):
            self.state_dim=parameter.sample_state_size
        elif(options.use_pitch):
            self.state_dim = parameter.pitch_state_size
        elif(options.use_spectrogram):
            self.state_dim = parameter.spectrogram_state_size
        else:
            self.state_dim = parameter.raw_data_state_size
        self.action_dim=parameter.action_size
        self.memory = deque(maxlen=2000)
        self.discount_factor = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.exploration_min = 0.01
        self.exploration_decay = 0.995
        if(options.use_dense):
            self.model = self.build_model()
        if(options.use_CNN_1D):
            self.model=self.build_CNN_model_1D()
        if(options.use_CNN_2D):
            self.model=self.build_CNN_model_2D()

        print(self.model.summary())

    def build_model(self):
        
        print("Neural Net for Deep-Q learning Model,Dense Network")
        model = Sequential()
        model.add(Dense(32, input_shape=(self.state_dim,), kernel_initializer='uniform', activation='relu'))
        model.add(Dense(24, activation='relu', kernel_initializer='uniform'))
        model.add(Dense(parameter.action_size, activation='softmax'))
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])
        return model
    
    def build_CNN_model_1D(self):
        print("Neural Net for Deep-Q learning Model, CNN 1D")
        model = Sequential()
        model.add(Conv1D(64, 3, activation='relu', kernel_initializer='uniform',input_shape=(self.state_dim, 1)))
        model.add(Conv1D(32, 3, kernel_initializer='uniform', activation='relu'))
        model.add(Flatten())
        model.add(Dense(parameter.action_size, activation='softmax'))
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer='sgd')
        return model

    def build_CNN_model_2D(self):
       
        model = Sequential()
        model.add(Conv2D(32, kernel_size=3, input_shape=(parameter.spectrogram_length, parameter.spectrogram_state_size, 1), activation='relu'))
        model.add(MaxPooling2D(2))
        model.add(Conv2D(32, kernel_size=3, activation='relu'))
        model.add(MaxPooling2D(2))
        model.add(Flatten())
        model.add(Dense(parameter.action_size, activation='softmax'))
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
        return model
       
        '''
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',kernel_initializer='uniform',
                         input_shape=(parameter.spectrogram_length, parameter.spectrogram_state_size, 1)))
        model.add(Conv2D(64, (3, 3),kernel_initializer='uniform', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, kernel_initializer='uniform',activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(parameter.action_size, activation='softmax'))
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
        return model
       ''' 

        '''
        model = Sequential()
        # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
        # this applies 32 convolution filters of size 3x3 each.
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(parameter.spectrogram_length, parameter.spectrogram_state_size, 1)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(parameter.action_size, activation='softmax'))

        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd)
        return model
        '''
    def replay_memory(self, state, action, reward, next_state, done):
        #print(state, action, reward, next_state, done)
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, env):
        # Epsilon-greedy agent policy
        if random.uniform(0, 1) < self.epsilon:
            # explore
            return np.random.choice(env.allowed_actions())
        else:
            # exploit on allowed actions
            state = env.state;
            actions_allowed = env.allowed_actions()
            Q_s = self.Q[state[0], state[1], actions_allowed]
            actions_greedy = actions_allowed[np.flatnonzero(Q_s == np.max(Q_s))]
            return np.random.choice(actions_greedy)

    def act(self, state,env):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(env.allowed_actions())
        else:
            act_values = self.model.predict(state)
            state = env.state;
            actions_allowed = env.allowed_actions()
            actions_allowed.sort()
            q_value_allowed = []
            for i in range(parameter.action_size):
                if i in actions_allowed:
                    q_value_allowed.append(act_values[0][i])
                else:
                    q_value_allowed.append(-100)
            return  np.argmax(q_value_allowed)  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.discount_factor * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
