'''
Created on Dec 21, 2017

@author: Marcus Ritter
'''

import random
import copy

class DataGenerator():
    
    def __init__(self, RANDOM_SEED, min_products, max_products, 
                min_sources, max_sources, min_product_quantity, max_product_quantity,
                min_source_distance, max_source_distance, min_source_inventory, max_source_inventory):

        # to use a random seed
        self.random_seed = RANDOM_SEED
        # maximum, minimum number of products
        self.max_products = max_products
        self.min_products = min_products
        # maximum, minimum number of sources
        self.max_sources = max_sources
        self.min_sources = min_sources
        # maximum, minimum quantity per product
        self.max_product_quantity = max_product_quantity
        self.min_product_quantity = min_product_quantity
        #maximum, minimum distance from source to customer
        self.max_source_distance = max_source_distance
        self.min_source_distance = min_source_distance
        # maximum, minimum quantity in stock per source
        self.max_source_inventory = max_source_inventory
        self.min_source_inventory = min_source_inventory

    'Generate a data set for training or testing'
    def generateDataSet(self, num_orders, seed_value):
        # generate random data list 
        data = []
        # to train with the same data set
        if(self.random_seed == True): random.seed(seed_value)
        else: random.seed()
        for _ in range(num_orders):
            data_set = DataSet(self.max_products, self.max_sources)
            data_set.generate_order(self.max_products, self.max_product_quantity,
                                    self.min_products, self.min_product_quantity)
            data_set.generate_sources(self.max_sources, self.max_source_distance,
                                      self.max_source_inventory, self.min_source_distance,
                                      self.min_source_inventory, self.max_products)
            data.append(data_set)
        return data
        
'Data set that generates a random customer order'    
class DataSet():
    
    def __init__(self, max_products, max_sources):
        self.number_products = 0
        # quantities of the products
        self.product_quantities = []
        for _ in range(max_products):
            self.product_quantities.append(0.0)
        # number of sources
        self.number_sources = 0
        # distance customer-source
        self.distances = []
        # stock situation
        stock_vector_template = []
        for _ in range(max_products):
            stock_vector_template.append(0.0)
        self.inventory_quantities = []
        for _ in range(max_sources):
            stock_vector_template_copy = copy.deepcopy(stock_vector_template)
            self.inventory_quantities.append(stock_vector_template_copy)
        # delivery
        delivery_vector_template = []
        for _ in range(max_products):
            delivery_vector_template.append(0.0)
        self.delivery = []
        for _ in range(max_sources):
            delivery_vector_template_copy = copy.deepcopy(delivery_vector_template)
            self.delivery.append(delivery_vector_template_copy)
    
    'Generates a random customer order according to the set parameters'
    def generate_order(self, max_products, max_product_quantity, min_products, min_product_quantity):
        self.number_products = random.randint(min_products, max_products)
        for i in range(self.number_products):
            prod_quantity = random.randint(min_product_quantity, max_product_quantity)
            # normalize the product quantity
            n_prod_quantity = self.normalize_pquantity(prod_quantity)
            self.product_quantities[i] = n_prod_quantity
        
    'Generates a random stock situation with sources, inventory quantity and distance to customer'
    def generate_sources(self, max_sources, max_source_distance, max_source_inventory, min_source_distance, min_source_inventory, max_products):
        self.number_sources = max_sources
        for _ in range(self.number_sources):
            distance = random.randint(min_source_distance, max_source_distance)
            # normalize the distance
            n_distance = self.normalize_distance(distance, max_source_distance)
            self.distances.append(n_distance)
        for i in range(self.number_sources):
            for j in range(max_products):    
                quantity = random.randint(min_source_inventory, max_source_inventory)
                # reduce the complexity of inventory stock
                if(quantity > 20): quantity = 20
                # normalize the inventory quantity
                n_quantity = self.normalize_inventory(quantity)
                n_quantity = round(n_quantity, 1)
                self.inventory_quantities[i][j] = n_quantity
    
    'normalizes the distance between customer and source'
    def normalize_distance(self, distance, max_source_distance):
        temp = max_source_distance / 100
        temp2 = distance / temp
        n_distance = temp2 * 0.01
        return n_distance
    
    'normalizes the product quantity'
    def normalize_pquantity(self, prod_quantity):
        n_prod_quantity = prod_quantity / 10
        return n_prod_quantity
    
    'normalizes the inventory stock'
    def normalize_inventory(self, quantity):
        n_quantity = (quantity / 10) - 1
        return n_quantity