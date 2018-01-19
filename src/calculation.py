'''
Created on Dec 21, 2017

@author: Marcus Ritter
'''

from data_generator import DataGenerator
from evaluation_environement import EvaluationEnvironment
from util import writeToCSV

import sys
import tensorflow as tf

if __name__ == '__main__':
    
    print("Parameters given: ",len(sys.argv))
    print(str(sys.argv))
    
    'test data generator seed'
    test_data_seed = None
    if(len(sys.argv) > 1): test_data_seed = int(sys.argv[1])
    else: test_data_seed = 1

    'training episodes parameter'
    training_episodes = None
    if(len(sys.argv) > 2): training_episodes = int(sys.argv[2])
    else: training_episodes = 10000
    
    '___ML PARAMETERS___'
    testing_episodes = 100
    epsilon = 0.9
    alpha = 0.001
    gamma = 0.01
    decay = 0.9
    replace_target_iter = 1
    memory_size = 500
    batch_size = 32
    STATICDATA = False
    RANDOM_SEED = True
    
    '___ENVIRONMENT PARAMETERS___'
    max_products = 10
    min_products = 1
    max_sources = 8
    min_sources = 1
    max_product_quantity = 10
    min_product_quantity = 1
    max_source_distance = 5000
    min_source_distance = 1
    max_source_inventory = 1000
    min_source_inventory = 10
    
    generator = DataGenerator(RANDOM_SEED, min_products, max_products, min_sources,
                              max_sources, min_product_quantity, max_product_quantity, min_source_distance,
                              max_source_distance, min_source_inventory, max_source_inventory)
    calculate_data = generator.generateDataSet(testing_episodes, test_data_seed)
    print("Test data generated.")
    
    max_rewards = []
    min_rewards = []
    
    for i in range(len(calculate_data)):
        
        print(str(i)+"%-100%")
        
        eval_env = EvaluationEnvironment(training_episodes, epsilon, alpha, 
                              gamma, decay, replace_target_iter, 
                              memory_size, batch_size, calculate_data[i], 
                              max_products, min_products, max_sources, 
                              min_sources, max_product_quantity, min_product_quantity, 
                              max_source_distance, min_source_distance, max_source_inventory, 
                              min_source_inventory)
        
        rewards = eval_env.train()
        eval_env = None
        max_rewards.append(max(rewards))
        min_rewards.append(min(rewards))
        
        tf.get_variable_scope().reuse_variables()
        
    print("Computed maximum, minimum test rewards.")
    
    path = "tests"
    writeToCSV(max_rewards, "test_max_rewards_"+str(test_data_seed), path)
    writeToCSV(min_rewards, "test_min_rewards_"+str(test_data_seed), path)
    print("Min, Max data written to Test folder.")
