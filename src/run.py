'''
Created on Dec 21, 2017

@author: Marcus Ritter
'''

from environment import Environment
from data_generator import DataGenerator
from util import writeToCSV
from util import createFolder
from util import writeParameterToFile
from util import simplify_loss

import sys

if __name__ == '__main__':
    
    print("Passed Parameters: ", len(sys.argv))
    print(str(sys.argv))
    
    '___ML PARAMETERS___'
    if(len(sys.argv) > 1): training_episodes = int(sys.argv[1])
    else: training_episodes = 10000
    testing_episodes = 100
    'vanilla optimizer values are used, only lr is passed'
    epsilon = 1.0
    epsilon_min = 0.1
    alpha = 0.001
    gamma = 0.01
    decay = 0.9
    replace_target_iter = 10
    memory_size = 200
    batch_size = 32
    STATICDATA = False
    RANDOM_SEED = True
    if(len(sys.argv) > 2): test_seed = int(sys.argv[2])
    else: test_seed = 1
    
    '___ENVIRONMENT PARAMETERS___'
    max_products = 2
    min_products = 1
    max_sources = 2
    min_sources = 1
    max_product_quantity = 10
    min_product_quantity = 1
    max_source_distance = 5000
    min_source_distance = 1
    max_source_inventory = 1000
    min_source_inventory = 10
    
    path = createFolder()
    writeParameterToFile(training_episodes, testing_episodes, epsilon, epsilon_min, alpha, 
                         gamma, decay, replace_target_iter, memory_size, batch_size, 
                         STATICDATA, RANDOM_SEED, max_products, min_products, 
                         max_sources, min_sources, max_product_quantity, min_product_quantity, 
                         max_source_distance, min_source_distance, max_source_inventory, 
                         min_source_inventory, path, test_seed)
    print("Parameter written to Test folder.")
    
    '___DATA GENERATION___'
    generator = DataGenerator(RANDOM_SEED, min_products, max_products, min_sources,
                              max_sources, min_product_quantity, max_product_quantity, min_source_distance,
                              max_source_distance, min_source_inventory, max_source_inventory)
    training_data = generator.generateDataSet(training_episodes, 1)
    test_data = generator.generateDataSet(testing_episodes, test_seed)
    print("Data generated.")
        
    '___ENVIRONMENT___'
    environment = Environment(training_episodes, testing_episodes, epsilon, epsilon_min, alpha, gamma, decay,
                              replace_target_iter, memory_size, batch_size, training_data,
                              test_data, max_products, min_products, max_sources, min_sources,
                              max_product_quantity, min_product_quantity, max_source_distance,
                              min_source_distance, max_source_inventory, min_source_inventory)
    
    '___TRAIN___'
    loss, train_rewards = environment.train()
    loss_ = simplify_loss(loss, 100)
    print("Agent trained.")
    writeToCSV(train_rewards, "train_rewards", path)
    writeToCSV(loss, "train_loss", path)
    print("Training data written to Test folder.")
        
    '___TEST___'
    reward = environment.test()
    print("Agent tested.")
    writeToCSV(reward, "test_rewards", path)
    print("Test data written to Test folder.")
