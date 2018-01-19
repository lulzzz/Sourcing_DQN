'''
Created on Jan 18, 2018

@author: Marcus Ritter
'''

from data_generator import DataGenerator
from operator import itemgetter
import math
from util import writeToCSV
from util import createFolder2
import matplotlib.pyplot as plt
import numpy as np


from util import writeParameterToFile
from util import simplify_loss

import sys

if __name__ == '__main__':
    
    '___SOURCING PARAMETERS___'
    num_sourcing_requests = 1000
    RANDOM_SEED = True
    
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
    
    '___DATA GENERATION___'
    generator = DataGenerator(RANDOM_SEED, min_products, max_products, min_sources,
                              max_sources, min_product_quantity, max_product_quantity, min_source_distance,
                              max_source_distance, min_source_inventory, max_source_inventory)
    test_data = generator.generateDataSet(num_sourcing_requests, 1)
    print("Data generated.")
    
    '___SOURCE REQUESTS___'
    rewards = []
    for i in range(num_sourcing_requests): 
        #print('sourcing request '+str(i))
        # sort sources by distance
        #print(test_data[i].distances)
        position_index = []
        for a in range(len(test_data[i].distances)):
            position_index.append([a, test_data[i].distances[a]])
        #print(position_index)
        sorted_sources_list = sorted(position_index, key=itemgetter(1))
        #print(sorted_sources_list)

        # check availability per source
        valid_source_id = None
        valid_source_found = False
        for b in range(test_data[i].number_sources):
            inventory_index = sorted_sources_list[b][0]
            #print(inventory_index)
            #print(test_data[i].inventory_quantities[inventory_index])

            #print(test_data[i].product_quantities)

            # check if this source is valid
            source_valid = False
            for c in range(test_data[i].number_products):
                # check product quantity
                n_prod_quantity = test_data[i].product_quantities[c]
                prod_quantity = n_prod_quantity * 10    
                #print(prod_quantity)

                # check inventory quantity
                n_inventory_quantity_quantity = test_data[i].inventory_quantities[inventory_index][c]
                inventory_quantity = n_inventory_quantity_quantity + 1
                inventory_quantity = inventory_quantity * 10
                #print(inventory_quantity)

                # check if possible to deliver
                remainder = inventory_quantity - prod_quantity
                if(remainder >= 0.0):
                    source_valid = True
                    #print(source_valid)
                    break
            
            # valid source was found
            if(source_valid == True):
                valid_source_id = inventory_index
                valid_source_found = True
                break

        # show result
        #print('valid source found: ', valid_source_found)
        #print('source_id: ', valid_source_id)

        reward = 0

        # if no valid source was found give fixed reward
        if(valid_source_found == False):
            reward = -10
        # else calculate the reward with the reward function
        else:
            # create the delivery vector
            #print(test_data[i].delivery)
            #print(test_data[i].product_quantities)

            for d in range(test_data[i].number_products):
                n_prod_quantity = test_data[i].product_quantities[d]
                prod_quantity = n_prod_quantity * 10
                test_data[i].delivery[valid_source_id][d] = prod_quantity
            
            #print(test_data[i].delivery)
            
            # calculate reward
            A1 = 4
            A2 = 2
            A3 = 1
            
            # number of used sources is 1
            used_sources = 1
            R1 = 0 - (math.pow(2, (used_sources-1)) - 1)
            if(R1 == 0): R1 = 1

            #print('R1: ', R1)
            
            # distance from customer to source
            R2 = 0
            d = test_data[i].distances[valid_source_id]
            #print(d)
            'until 1250km +reward'
            R2 = 1 - (d / 0.25)
            
            #print('R2: ', R2)

            # remainder of stock
            R3 = 0
            stock_situation = []
            for i in range(max_sources):
                for j in range(max_products):
                    delivery = test_data[i].delivery[i][j]
                    if(delivery > 0):
                        # denormalize value
                        n_inventory_stock = test_data[i].inventory_quantities[i][j]
                        inventory_stock = n_inventory_stock + 1
                        inventory_stock = inventory_stock * 10
                        stock = inventory_stock - delivery
                        stock_situation.append(stock)
            
            for i in range(len(stock_situation)):
                r = (math.pow(2, (stock_situation[i]-1)) - 1)
                if(r < 0): r = r * 2
                if(r > 0): r = r / 4
                if(r > 2): r = 2
                R3 = R3 + r
            R3 = R3 / len(stock_situation)
            
            #print('R3: ', R3)

            # combined reward
            R = A1 * R1 + A2 * R2 + A3 * R3

            #print('R: ', R)

            reward = R
        
        rewards.append(reward)
    
    # show all request rewards
    #print(rewards)

    '___WRITE RESULTS TO TEST FOLDER___'
    path = createFolder2()
    writeToCSV(rewards, "test_rewards", path)

    '___PLOT TEST DATA___'
    plt.plot(np.arange(len(rewards)), rewards, 'b.')
    plt.grid()
    plt.ylabel('reward')
    plt.xlabel('request number')
    plt.title('vanilla sourcing test results')
    plt.show()