# -*- coding: utf-8 -*-
from Environment import  Environment
from ExtractFeatures import Extract_Features
from GlobalVariables import GlobalVariables
from DQNAgent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt
#import pylab
import sys

import matplotlib.pylab as plt

Extract=Extract_Features

options=GlobalVariables #To access global variables from GlobalVariable.py
parameter=GlobalVariables #To access parameters from GlobalVariables.py
samples=Extract_Features #To access the member functions of the ExtractFeatures class
grid_size=GlobalVariables #To access the size of grid from Global Variables.py

env = Environment(grid_size.nRow,grid_size.nCol)
agent = DQNAgent(env)

list_Number_of_Iterations=[]
list_Reward=[]

for i in range(parameter.how_many_times):
    print("************************************************************************************")
    print("Iteration",i+1)
    Number_of_Iterations=[]
    Number_of_Episodes=[]
    reward_List = []
    #filename = str(grid_size.nRow) + "X" + str(grid_size.nCol) + "_Experiment.txt"
    for episode in range(1,parameter.Number_of_episodes+1):
        #file = open(filename, 'a')
        #done = False
        #state,goal_state,wall = env.reset()
        state,goal_state = env.reset()
        if (options.use_samples and options.use_dense):
            state=samples.Extract_Samples(state[0],state[1])
            samples_goal = samples.Extract_Samples(goal_state[0],goal_state[1])
            state = np.reshape(state, [1, parameter.sample_state_size])

        if (options.use_samples and options.use_CNN_1D):
            state=samples.Extract_Samples(state[0],state[1])
            samples_goal = samples.Extract_Samples(goal_state[0],goal_state[1])
            state = np.reshape(state, [1, parameter.sample_state_size,1])


        if (options.use_pitch and options.use_dense):
            state = samples.Extract_Pitch(state[0], state[1])
            samples_goal = samples.Extract_Pitch(goal_state[0],goal_state[1])
            state = np.reshape(state, [1, parameter.pitch_state_size])

        if (options.use_pitch and options.use_CNN_1D):
            state = samples.Extract_Pitch(state[0], state[1])
            samples_goal = samples.Extract_Pitch(goal_state[0],goal_state[1])
            state = np.reshape(state, [1, parameter.pitch_state_size,1])

        if (options.use_spectrogram):
            state = samples.Extract_Spectrogram(state[0], state[1])
            samples_goal = samples.Extract_Spectrogram(goal_state[0],goal_state[1])
            state = np.reshape(state, [1,parameter.spectrogram_length, parameter.spectrogram_state_size, 1])
        
        iterations=0
        Number_of_Episodes.append(episode)
        for time in range(parameter.timesteps):
        #done=False
        #while True:
        #while not done:
            #print("Two")
            iterations+=1
            action = agent.act(state,env)
            #
            next_state, reward, done = env.step(action,samples_goal)

            if(options.use_samples and options.use_dense):
                next_state = np.reshape(next_state, [1, parameter.sample_state_size])

            if(options.use_samples and options.use_CNN_1D):
                next_state = np.reshape(next_state, [1, parameter.sample_state_size,1])

            if (options.use_pitch and options.use_dense):
                next_state = np.reshape(next_state, [1, parameter.pitch_state_size])

            if (options.use_pitch and options.use_CNN_1D):
                next_state = np.reshape(next_state, [1, parameter.pitch_state_size,1])

            if (options.use_spectrogram):
                next_state = np.reshape(next_state, [1,parameter.spectrogram_length, parameter.spectrogram_state_size,1])

            agent.replay_memory(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
            if len(agent.memory) > parameter.batch_size:
                agent.replay(parameter.batch_size)
        Number_of_Iterations.append(iterations)
        reward_List.append(reward)
        print("episode: {}/{}, iteration: {}, reward {}".format(episode, parameter.Number_of_episodes, iterations, reward))

    #print(Number_of_Episodes)
    #print(Number_of_Iterations)
    # file.write("Episode = " + str(Number_of_Episodes))
    # file.write(str(Number_of_Iterations))
    # file.write('\n')
    #file.close()
    
    list_Number_of_Iterations.append(Number_of_Iterations)
    list_Reward.append(reward_List)
    #print(list)

    percentage_of_successful_episodes = (sum(reward_List) / parameter.Number_of_episodes) * 100
    print("Percentage of Successful Episodes at Iteration {} is {} {}".format(i+1,percentage_of_successful_episodes, '%'))

    fig = plt.figure()
    fig.suptitle('Q-Learning', fontsize=12)
    title=str(grid_size.nRow) + "X" + str(grid_size.nCol) + '_'+ str(i+1)
    fig.suptitle(title, fontsize=12)
    plt.plot(np.arange(len(Number_of_Episodes)), Number_of_Iterations)
    #plt.plot(np.arange(len(Number_of_Episodes)), reward_List)
    plt.ylabel('Number of Iterations')
    #plt.ylabel('Reward')
    plt.xlabel('Episode Number')
    filename=title+'.png'
    #plt.savefig(filename)
    plt.show(block=False)
    plt.pause(3)
    plt.close()
    print("************************************************************************************")

mu_Iteration=np.mean(list_Number_of_Iterations, axis=0)
std_Iteration=np.std(list_Number_of_Iterations, axis=0)

mu_Reward=np.mean(list_Reward, axis=0)
std_Reward=np.std(list_Reward, axis=0)

#print("Mean",mu_Iteration)
#print("Std",std_Iteration)

Episode_Number = []
for i in range(1,len(list_Number_of_Iterations[0])+1):
    Episode_Number.append(i)

startpos=options.start
if(startpos==0):
    start_option="Fixed"
else:
    start_option="Random"

goalpos=options.goal
if(goalpos==0):
    goal_option="Fixed"
else:
    goal_option="Random"

# plt.plot(Episode_Number, mu, '-b', label='Mean')
# plt.plot(Episode_Number, std, '-r', label='Standard Deviation')
# plt.legend(loc='upper right')
# plt.ylim(0, max(np.max(mu),np.max(std))+1)
# plt.xlim(1, np.max(Episode_Number)+1)
# plt.xlabel('Episode Number')
# plt.ylabel('Reward')
# filename_curve='./Learning_Curves/'+str(grid_size.nRow)+'X'+str(grid_size.nCol)+'_'+str(parameter.how_many_times)+'_times_'+'start_'+start_option+ '_goal_'+goal_option+'.png'
# title='Grid Size = '+str(grid_size.nRow) + 'X'+str(grid_size.nCol)+', Start =' + start_option + ', Goal =' + goal_option + ', Experiment Carried out = '+ str(parameter.how_many_times)+' times'
# plt.suptitle(title, fontsize=12)
# plt.savefig(filename_curve)
# plt.show()
# #plt.show(block=False)
# #plt.pause(3)
# #plt.close()

filename_curve='./Learning_Curves/'+str(grid_size.nRow)+'X'+str(grid_size.nCol)+'_'+str(parameter.how_many_times)+'_times_'+'start_'+start_option+ '_goal_'+goal_option+'.png'

time = np.arange(np.max(Episode_Number))


# plot it!
#fig, ax = plt.subplots(1)
plt.plot(time, mu_Iteration, lw=2, alpha=0.5,label='Mean Iteration per Episdoe', color='red')
plt.plot(time,mu_Reward,color='green',label='Mean Reward per Episode')
plt.fill_between(time, mu_Iteration-std_Iteration, mu_Iteration+std_Iteration, facecolor='blue', alpha=0.3)
plt.legend(loc='upper right')
plt.xlabel('Number of Episodes')
#plt.ylabel('Mean Iteration')
plt.grid()
plt.savefig(filename_curve)
plt.show()


