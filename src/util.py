'''
Created on Dec 18, 2017

@author: Marcus Ritter
'''

import collections
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import shutil

'normalizes the distance between customer and source'
def normalize_distance(distance, max_source_distance):
    one_percent = max_source_distance / 100
    one_percent_d = distance / one_percent
    normalized_distance = one_percent_d * 0.01
    return normalized_distance

'normalizes the product quantity'
def normalize_product_quantity(prod_quantity):
    normalized_prod_quantity = prod_quantity / 10
    return normalized_prod_quantity

'normalizes the inventory quantity'
def normalize_inventory_quantity(inventory_quantity):
    normalized_quantity = (inventory_quantity / 10) - 1
    return normalized_quantity




def createFolder2():
    path = 'tests\\'
    folder_name = 'vanilla_sourcing'
    path = path + folder_name
    if(not os.path.exists(path)):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)
    return path

'creates a new folder for the test data inside the test folder'
def createFolder():
    path = 'tests\\'
    folder_list = os.listdir('tests\\')
    folder_name = 0
    name_is_valid = False
    # check if the folder name is valid
    while(name_is_valid != True):
        if(check_foldername(folder_list, folder_name) == True):
            folder_name = folder_name + 1
        else:
            name_is_valid = True
            break
    path = path + str(folder_name)
    if(not os.path.exists(path)):
        os.makedirs(path)
    return path

'check folders in directory'
def check_foldername(folder_list, test_name):
    is_existing = False
    for i in range(len(folder_list)):
        if(folder_list[i] == test_name):
            is_existing = True
            break
    return is_existing

'write a data list to a csv file'
def writeToCSV(data_list, file_name, path_):
    nrlist = []
    for i in range(len(data_list)):
        nrlist.append(i+1)
    dictionary = {}
    for i in range(len(data_list)):
        dictionary[nrlist[i]] = data_list[i]
    # create the file path
    path = path_
    data_type = '.csv'
    file_path = path+'/'+file_name+data_type
    # create the column titles
    column_titles = 'nr,value'
    # sort the dictionary
    odict = collections.OrderedDict(sorted(dictionary.items()))
    # write to file
    csv = open(file_path, 'w')
    csv.write(column_titles)
    for key in odict.keys():
        nr = key
        value =  odict[key]
        row = '\n' + str(nr) + ',' + str(value)
        csv.write(row)
        
'writes the parameters to a txt file'
def writeParameterToFile(environment_parameters,training_episodes,testing_episodes,epsilon,epsilon_min,alpha,gamma,
    decay,replace_target_iter,memory_size,batch_size,use_seed,
    path_,testseed):
    path = path_
    data_type = '.txt'
    file_path = path+'/'+"parameter"+data_type
    file = open(file_path, 'w')
    str1 = " training_episodes: "+str(training_episodes)
    str2 = "\n testing_episodes: "+str(testing_episodes)
    str3 = "\n epsilon: "+str(epsilon)
    str23 = "\n epsilon_min: "+str(epsilon_min)
    str4 = "\n alpha: "+str(alpha)
    str5 = "\n gamma: "+str(gamma)
    str6 = "\n decay: "+str(decay)
    str7 = "\n replace_target_iter: "+str(replace_target_iter)
    str8 = "\n memory_size: "+str(memory_size)
    str9 = "\n batch_size: "+str(batch_size)
    str11 = "\n RANDOM_SEED: "+str(use_seed)
    str12 = "\n max_products: "+str(environment_parameters.max_products)
    str13 = "\n min_products: "+str(environment_parameters.min_products)
    str14 = "\n max_sources: "+str(environment_parameters.max_sources)
    str15 = "\n min_sources: "+str(environment_parameters.min_sources)
    str16 = "\n max_product_quantity: "+str(environment_parameters.max_product_quantity)
    str17 = "\n min_product_quantity: "+str(environment_parameters.min_product_quantity)
    str18 = "\n max_source_distance: "+str(environment_parameters.max_source_distance)
    str19 = "\n min_source_distance: "+str(environment_parameters.min_source_distance)
    str20 = "\n max_source_inventory: "+str(environment_parameters.max_source_inventory)
    str21 = "\n min_source_inventory: "+str(environment_parameters.min_source_inventory)
    str22 = "\n test seed: "+str(testseed)
    file.write(str1)
    file.write(str2)
    file.write(str3)
    file.write(str23)
    file.write(str4)
    file.write(str5)
    file.write(str6)
    file.write(str7)
    file.write(str8)
    file.write(str9)
    file.write(str11)
    file.write(str12)
    file.write(str13)
    file.write(str14)
    file.write(str15)
    file.write(str16)
    file.write(str17)
    file.write(str18)
    file.write(str19)
    file.write(str20)
    file.write(str21)
    file.write(str22)
    file.close()
    
'writes the results of the evaluation to file'
def writeResultsToFile(testing_episodes, average_reward, percentage, path_):
    path = path_
    data_type = '.txt'
    file_path = path+'/'+"results"+data_type
    file = open(file_path, 'w')
    str1 = "Number of test episodes: "+str(testing_episodes)
    str2 = "\nAverage Testing Reward: "+str(average_reward)
    str3 = "\nAccuracy: "+str(percentage)
    file.write(str1)
    file.write(str2)
    file.write(str3)
    file.close()
    
'read parameter file'
def readParameterFile(path):
    file = open(path+"parameter.txt")
    string = file.readline()
    strings = string.split(':')
    strings2 = strings[1].split()
    return strings2[0]
        
'reads the train loss'
def readTrainLoss(path):
    cost = []
    file = open(path+"train_loss.csv")
    reader = csv.reader(file)
    for line in reader:
        if(line[1]!='value'): cost.append(line[1])
    return cost

'reads the train reward'
def readTrainReward(path):
    train_reward = []
    file = open(path+"train_rewards.csv")
    reader = csv.reader(file)
    for line in reader:
        if(line[1]!='value'): train_reward.append(line[1])
    return train_reward

'reads the test reward'
def readTestReward(path):
    test_reward = []
    file = open(path+"test_rewards.csv")
    reader = csv.reader(file)
    for line in reader:
        if(line[1]!='value'): test_reward.append(line[1])
    return test_reward

'reads the maximum test reward'
def readTestMaxReward(path):
    test_max_reward = []
    file = open(path)
    reader = csv.reader(file)
    for line in reader:
        if(line[1]!='value'): test_max_reward.append(line[1])
    return test_max_reward

'reads the minimum test reward'
def readTestMinReward(path):
    test_min_reward = []
    file = open(path)
    reader = csv.reader(file)
    for line in reader:
        if(line[1]!='value'): test_min_reward.append(line[1])
    return test_min_reward

'plot the cost function, cost per sourcing step'    
def plotCostFunction(data_list):
    # use log scale for y
    plt.semilogy(np.arange(len(data_list)), data_list)
    plt.grid()
    plt.ylabel('Cost')
    plt.xlabel('Sourcing Step')
    plt.title('Training Process Cost Function')
    plt.show()

'plot the rewards from the training phase'
def plotTrainingRewards(data_list):
    plt.plot(np.arange(len(data_list)), data_list, 'b.')
    plt.grid()
    plt.ylabel('Sourcing Reward')
    plt.xlabel('Order')
    plt.title('Training Process Sourcing Rewards')
    plt.show()

'plot the rewards from the testing phase'
def plotTestRewards(data_list):
    plt.plot(np.arange(len(data_list)), data_list, 'b.')
    plt.grid()
    plt.ylabel('Sourcing Reward')
    plt.xlabel('Order')
    plt.title('Testing Process Sourcing Rewards')
    plt.show()

'plot the rewards of the testing phase and the max possible reward and the minimum'
def plotTestResults(test_rewards, min_rewards, max_rewards, training_episodes):
    plt.plot(np.arange(len(min_rewards)), min_rewards, 'ro', label='min reward')
    plt.plot(np.arange(len(max_rewards)), max_rewards, 'gs-', label='max reward', lw=4)
    plt.plot(np.arange(len(test_rewards)), test_rewards, 'bD', label='trained model reward')
    plt.grid()
    plt.legend()
    plt.title('Sourcing Test Results (Training Orders='+str(training_episodes)+')')
    plt.ylabel('Sourcing Reward')
    plt.xlabel('Order')
    plt.show()

'simplify the loss data'   
def simplify_loss(loss, step_size):
    counter = 0
    limit = step_size
    loss_ = []
    for i in range(len(loss)):
        counter = counter + 1
        if(counter >= limit):
            loss_.append(loss[i])
            counter = 0
    return loss_