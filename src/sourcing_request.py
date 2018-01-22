'''
Created on Jan 20, 2018

@author: Marcus Ritter
'''

from util import normalize_distance
from util import normalize_product_quantity
from util import normalize_inventory_quantity
import copy
import random

class SourcingRequest():

    def __init__(self, max_products, min_products, max_product_quantity, min_product_quantity, max_sources, 
    max_source_distance, min_source_distance, max_source_inventory, min_source_inventory):

        # defines the complexity limit for the inventory stock
        self.inventory_complexity_limit = 10

        self.max_products = max_products
        self.min_products = min_products
        self.max_product_quantity = max_product_quantity
        self.min_product_quantity = min_product_quantity
        self.max_sources = max_sources
        self.max_source_distance = max_source_distance
        self.min_source_distance = min_source_distance
        self.max_source_inventory = max_source_inventory
        self.min_source_inventory = min_source_inventory

        self.number_products = 0

        self.product_quantities = []
        for _ in range(self.max_products):
            self.product_quantities.append(0.0)

        self.number_sources = 0

        self.distances = []

        stock_vector_template = []
        for _ in range(self.max_products):
            stock_vector_template.append(0.0)
        self.inventory_quantities = []
        for _ in range(self.max_sources):
            stock_vector_template_copy = copy.deepcopy(stock_vector_template)
            self.inventory_quantities.append(stock_vector_template_copy)

        delivery_vector_template = []
        for _ in range(self.max_products):
            delivery_vector_template.append(0.0)
        self.delivery = []
        for _ in range(self.max_sources):
            delivery_vector_template_copy = copy.deepcopy(delivery_vector_template)
            self.delivery.append(delivery_vector_template_copy)

    'Generate an order with n products and m product quantity'
    def generate_order(self):
        self.number_products = random.randint(self.min_products, self.max_products)

        for i in range(self.number_products):
            product_quantity = random.randint(self.min_product_quantity, self.max_product_quantity)

            # normalize product quantity
            normalized_product_quantity = normalize_product_quantity(product_quantity)
            self.product_quantities[i] = normalized_product_quantity

    'Generate n sources with different products on stock and distance to customer'
    def generate_sources(self):
        self.number_sources = self.max_sources

        for _ in range(self.number_sources):
            distance = random.randint(self.min_source_distance, self.max_source_distance)
            # normalize distance
            normalized_distance = normalize_distance(distance, self.max_source_distance)
            self.distances.append(normalized_distance)

        for i in range(self.number_sources):
            for j in range(self.max_products):
                inventory_quantity = random.randint(self.min_source_inventory, self.max_source_inventory)

                # stock limit
                # reduce complexity
                if(inventory_quantity >= self.inventory_complexity_limit):
                    inventory_quantity = self.inventory_complexity_limit

                # normalize inventory quantity
                normalized_quantity = normalize_inventory_quantity(inventory_quantity)
                normalized_quantity = round(normalized_quantity, 1)
                self.inventory_quantities[i][j] = normalized_quantity