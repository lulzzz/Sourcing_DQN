'''
Created on Jan 18, 2018

@author: Marcus Ritter

Implementation of the vanilla sourcing strategy one consignement today.
'''

from data_generator import DataGenerator
from operator import itemgetter
import math
from util import write_list_to_csv
from util import create_test_folder
import matplotlib.pyplot as plt
import numpy as np
from parameter import Parameters
from util import inventory_quantity_real_value
from util import product_quantity_real_value

if __name__ == '__main__':
    
    '___SOURCING PARAMETERS___'
    num_sourcing_requests = 100
    use_seed = True
    test_seed = 1
    failed_sourcing_reward = -10

    debug_print = False
    plot_results = False
    
    '___ENVIRONMENT PARAMETERS___'
    environment_parameters = Parameters()
    
    '___DATA GENERATION___'
    generator = DataGenerator(use_seed, environment_parameters)
    test_data = generator.generateDataSet(num_sourcing_requests, test_seed)
    if(debug_print == True):
        print("Data generated.")
    
    '___SOURCE REQUESTS___'
    rewards = []
    results = []
    for i in range(num_sourcing_requests): 
        if(debug_print == True):
            print('sourcing request '+str(i))
        # sort sources by distance
        if(debug_print == True):
            print(test_data[i].distances)
        position_index = []
        for a in range(len(test_data[i].distances)):
            position_index.append([a, test_data[i].distances[a]])
        if(debug_print == True):
            print(position_index)
        sorted_sources_list = sorted(position_index, key=itemgetter(1))
        if(debug_print == True):
            print(sorted_sources_list)

        # check availability per source
        valid_source_id = None
        valid_source_found = False
        for b in range(test_data[i].number_sources):
            inventory_index = sorted_sources_list[b][0]
            if(debug_print == True):
                print('\nsource number: '+str(inventory_index))
                print('stock: '+str(test_data[i].inventory_quantities[inventory_index]))
                print('product quantities: '+str(test_data[i].product_quantities))

            # check if this source is valid
            source_valid = True
            for c in range(test_data[i].number_products):
                # check product quantity
                n_prod_quantity = test_data[i].product_quantities[c]
                prod_quantity = product_quantity_real_value(n_prod_quantity)    
                if(debug_print == True):
                    print('product('+str(c)+'): '+str(prod_quantity))

                # check inventory quantity
                n_inventory_quantity_quantity = test_data[i].inventory_quantities[inventory_index][c]
                inventory_quantity = inventory_quantity_real_value(n_inventory_quantity_quantity)
                if(debug_print == True):
                    print('stock('+str(c)+'): '+str(inventory_quantity))

                # check if possible to deliver
                remainder = inventory_quantity - prod_quantity
                if(remainder < 0.0):
                    source_valid = False
                    if(debug_print == True):
                        print(source_valid)
                    break
            
            # valid source was found
            if(source_valid == True):
                valid_source_id = inventory_index
                valid_source_found = True
                break
            else:
                valid_source_found = False

        # show result
        if(debug_print == True):
            print('valid source found: ', valid_source_found)
            print('source_id: ', valid_source_id)

        reward = 0

        # if no valid source was found give fixed reward and set result
        if(valid_source_found == False):
            reward = failed_sourcing_reward
            result_value = False
        # else calculate the reward with the reward function
        else:
            # create the delivery vector
            if(debug_print == True):
                print(test_data[i].delivery)
                print(test_data[i].product_quantities)

            for d in range(test_data[i].number_products):
                n_prod_quantity = test_data[i].product_quantities[d]
                prod_quantity = n_prod_quantity * 10
                test_data[i].delivery[valid_source_id][d] = prod_quantity
            
            if(debug_print == True):
                print(test_data[i].delivery)
            
            # calculate reward
            # parameter
            A1 = 4
            A2 = 2
            A3 = 1
            
            # number of used sources is 1
            used_sources = 1
            R1 = 0 - (math.pow(2, (used_sources-1)) - 1)
            if(R1 == 0): R1 = 1

            if(debug_print == True):
                print('R1: ', R1)
            
            # distance from customer to source
            R2 = 0
            d = test_data[i].distances[valid_source_id]
            if(debug_print == True):
                print(d)
            'until 1250km +reward'
            R2 = 1 - (d / 0.25)
            
            if(debug_print == True):
                print('R2: ', R2)

            # remainder of stock
            R3 = 0
            stock_situation = []
            for x in range(environment_parameters.max_sources):
                for y in range(environment_parameters.max_products):
                    delivery = test_data[i].delivery[x][y]
                    if(delivery > 0.0):
                        # denormalize value
                        n_inventory_stock = test_data[i].inventory_quantities[x][y]
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
            
            if(debug_print == True):
                print('R3: ', R3)

            # combined reward
            R = A1 * R1 + A2 * R2 + A3 * R3

            if(debug_print == True):
                print('R: ', R)

            reward = R
            result_value = True
        
        rewards.append(reward)
        results.append(result_value)
    
    # show all request rewards
    if(debug_print == True):
        print(rewards)
        print(results)

    '___SAVE THE TEST RESULTS___'
    # create test folder
    path = create_test_folder('one_consignement_today')
    # write test rewards to file
    write_list_to_csv(rewards, "sourcing_rewards", path)
    if(debug_print == True):
        print('Test results saved.')

    '___PLOT TEST RESULTS___'
    successfull_sourcings = []
    for i in range(len(rewards)):
        if(results[i] == True):
            successfull_sourcings.append(rewards[i])
        else:
            successfull_sourcings.append(-20)

    failed_sourcings = []
    for i in range(len(rewards)):
        if(results[i] == False):
            failed_sourcings.append(rewards[i])
        else:
            failed_sourcings.append(-20)

    fig = plt.figure()
    plt.axis([0,100,-11,10])
    plt.plot(np.arange(len(rewards)), successfull_sourcings, 'g+')
    plt.plot(np.arange(len(rewards)), failed_sourcings, 'r+')
    plt.grid()
    plt.ylabel('reward')
    plt.xlabel('sourcing requests')
    plt.title('One Consignement Today Sourcing Results')
    fig.savefig(path+'\\plot_rewards.png')
    if(plot_results == True):
        plt.show()