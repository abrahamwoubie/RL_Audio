import numpy as np
import pylab
from ExtractFeatures import Extract_Features
from GlobalVariables import GlobalVariables
options=GlobalVariables #To access global variables from GlobalVariable.py
parameter=GlobalVariables #To access parameters from GlobalVariables.py
samples=Extract_Features #To access the member functions of the ExtractFeatures class
grid_size=GlobalVariables #To access the size of grid from Global Variables.py


mu=np.mean(list, axis=0)
std=np.std(list, axis=0)

Episode_Number = []
for i in range(1,len(list[0])+1):
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

print("Start Option",start_option)
pylab.plot(Episode_Number, mu, '-b', label='Mean')
pylab.plot(Episode_Number, std, '-r', label='Standard Deviation')
pylab.legend(loc='upper right')
pylab.ylim(0, max(np.max(mu),np.max(std))+1)
pylab.xlim(1, np.max(Episode_Number)+1)
pylab.xlabel('Episode Number')
pylab.ylabel('Iteration')
filename=str(grid_size.nRow)+'X'+str(grid_size.nCol)+'_'+str(parameter.how_many_times)+'_times_'+'start_'+start_option+ 'goal_'+goal_option+'.png'
title='Grid Size = '+str(grid_size.nRow) + 'X'+str(grid_size.nCol)+', Start =' + start_option + ', Goal =' + goal_option + ', Experiment Carried out = '+ str(parameter.how_many_times)+' times'
pylab.suptitle(title, fontsize=12)
#pylab.savefig(filename)
pylab.show()
