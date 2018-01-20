'''
Created on Dec 21, 2017

@author: Marcus Ritter
'''

import random
from sourcing_request import SourcingRequest

class DataGenerator():

    def __init__(self, use_seed, environment_parameters):

        # to use a random seed
        self.use_seed = use_seed
        # maximum, minimum number of products
        self.max_products = environment_parameters.max_products
        self.min_products = environment_parameters.min_products
        # maximum, minimum number of sources
        self.max_sources = environment_parameters.max_sources
        self.min_sources = environment_parameters.min_sources
        # maximum, minimum quantity per product
        self.max_product_quantity = environment_parameters.max_product_quantity
        self.min_product_quantity = environment_parameters.min_product_quantity
        # maximum, minimum distance from source to customer
        self.max_source_distance = environment_parameters.max_source_distance
        self.min_source_distance = environment_parameters.min_source_distance
        # maximum, minimum quantity in stock per source
        self.max_source_inventory = environment_parameters.max_source_inventory
        self.min_source_inventory = environment_parameters.min_source_inventory

    'Generate a data set with n customer orders'
    def generateDataSet(self, num_orders, generator_seed):
        data_set = []

        # use specific seed
        if(self.use_seed == True):
            random.seed(generator_seed)
        # generate random data
        else:
            random.seed()

        for _ in range(num_orders):
            sourcing_request = SourcingRequest(self.max_products, self.min_products, self.max_product_quantity,
                                               self.min_product_quantity, self.max_sources, self.max_source_distance,
                                               self.min_source_distance, self.max_source_inventory, self.min_source_inventory)
            sourcing_request.generate_order()
            sourcing_request.generate_sources()
            data_set.append(sourcing_request)

        return data_set