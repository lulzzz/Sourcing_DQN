'''
Created on Dec 21, 2017

@author: Marcus Ritter
'''

from util import plotCostFunction
from util import plotTrainingRewards
from util import plotTestRewards
from util import plotTestResults
from util import readTrainLoss
from util import readTrainReward
from util import readTestReward
from util import readTestMaxReward
from util import readTestMinReward
from util import readParameterFile
from util import writeResultsToFile

import matplotlib
import sys

if __name__ == '__main__':
    
    print("Passed Parameters: ", len(sys.argv))
    print(str(sys.argv))
    
    '___LOAD DATA___'
    test_folder_name = 'tests/'
    if(len(sys.argv) > 1): folder_name = str(sys.argv[1])+'/'
    else: folder_name = '0/'
    path = test_folder_name + folder_name
    training_episodes = readParameterFile(path)
    cost = readTrainLoss(path)
    training_rewards = readTrainReward(path)
    testing_rewards = readTestReward(path)
    #min_rewards = readTestMinReward("tests/test_min_rewards_1.csv")
    #max_rewards = readTestMaxReward("tests/test_max_rewards_1.csv")
    
    '___PLOT___'
    #matplotlib.rcParams.update({'font.size':18})
    #plotCostFunction(cost)
    #plotTrainingRewards(training_rewards)
    #plotTestRewards(testing_rewards)
    #plotTestResults(testing_rewards, min_rewards, max_rewards, training_episodes)
    
    '___STATISTIC___'
    sumr = 0
    for i in range(len(testing_rewards)):
        sumr = sumr + float(testing_rewards[i])
    average = sumr/len(testing_rewards)
    print(average)
    #sum_reward = 0
    #for i in range(len(testing_rewards)):
    #    sum_reward = sum_reward + float(testing_rewards[i])
    #sum_max_rewards = 0
    #for i in range(len(max_rewards)):
    #    sum_max_rewards = sum_max_rewards + float(max_rewards[i])
    #one_percent = sum_max_rewards / 100
    #percentage = sum_reward / one_percent
    #writeResultsToFile(len(testing_rewards), average, percentage, path)
    #print("Results written to test folder.")