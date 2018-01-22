'''
Created on Dec 21, 2017

@author: Marcus Ritter
'''

from environment import Environment
from data_generator import DataGenerator
from util import write_list_to_csv
from util import create_test_folder
from util import writeParameterToFile
from util import simplify_loss
from parameter import Parameters
import sys
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    
    '___ML PARAMETERS___'
    training_episodes = 10000
    testing_episodes = 100
    epsilon = 1.0
    epsilon_min = 0.1
    alpha = 0.001
    gamma = 0.01
    decay = 0.9
    replace_target_iter = 1
    memory_size = 200
    batch_size = 32
    use_seed = True
    training_seed = 1
    test_seed = 1

    debug_print = True

    '___ENVIRONMENT PARAMETERS___'
    environment_parameters = Parameters()
    
    '___CREATE TEST FOLDER___'
    path = create_test_folder('ai')
    writeParameterToFile(environment_parameters, training_episodes, testing_episodes, epsilon, epsilon_min, alpha, 
                         gamma, decay, replace_target_iter, memory_size, batch_size, 
                         use_seed, path, test_seed)
    if(debug_print == True):
        print("Parameter written to Test folder.")
    
    '___DATA GENERATION___'
    generator = DataGenerator(use_seed, environment_parameters)
    training_data = generator.generateDataSet(training_episodes, training_seed)
    test_data = generator.generateDataSet(testing_episodes, test_seed)
    if(debug_print == True):
        print("Data generated.")

    '___CREATE ENVIRONMENT___'
    environment = Environment(training_episodes, testing_episodes, epsilon, epsilon_min, alpha, gamma, decay,
                              replace_target_iter, memory_size, batch_size, training_data,
                              test_data, environment_parameters)
    
    '___TRAIN___'
    loss, train_rewards = environment.train()
    loss_ = simplify_loss(loss, 100)
    if(debug_print == True):
        print("Agent trained.")
    write_list_to_csv(train_rewards, "rewards", path)
    write_list_to_csv(loss, "loss", path)
    if(debug_print == True):
        print("Training data written to Test folder.")

    '___TEST___'
    path = create_test_folder('sourcing_dqn')
    sum_rewards, test_rewards = environment.test()
    if(debug_print == True):
        print("Agent tested.")
    write_list_to_csv(sum_rewards, "sum_rewards", path)
    write_list_to_csv(test_rewards, "test_rewards", path)
    if(debug_print == True):
        print("Test data written to Test folder.")

    fig = plt.figure()
    plt.axis([0,100,-11,10])
    plt.plot(np.arange(len(test_rewards)), test_rewards, 'b+')
    plt.grid()
    plt.ylabel('reward')
    plt.xlabel('sourcing requests')
    plt.title('DQN Sourcing Results')
    fig.savefig(path+'\\plot_rewards.png')
    plt.show()

    # compute average reward
    sumr = 0
    for i in range(len(test_rewards)):
        sumr = sumr + float(test_rewards[i])
    average = sumr/len(test_rewards)
    print(average)