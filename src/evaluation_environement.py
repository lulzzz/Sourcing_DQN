'''
Created on Dec 21, 2017

@author: Marcus Ritter
'''

import numpy as np
import tensorflow as tf
import math
import copy

from state import State

class EvaluationEnvironment():

    def __init__(self, train_episodes, epsilon, alpha, gamma, decay,
                 replace_target_iter, memory_size, batch_size, training_data,
                 max_products, min_products, max_sources, min_sources,
                 max_product_quantity, min_product_quantity, max_source_distance,
                 min_source_distance, max_source_inventory, min_source_inventory):
        
        '___Reinforcement Learning Variables___'
        # valid actions in the current episode
        self.ACTIONS = []
        self.ACTIONS_INDEX = []
        # all possible actions
        self.ALL_ACTIONS = []
        self.ALL_ACTIONS_INDEX = []
        # number of all possible actions
        self.n_actions = 0
        # number of features, inputs for the states
        self.n_features = 0
        # greedy policy
        self.EPSILON = epsilon
        # learning rate
        self.ALPHA = alpha   
        # discount factor
        self.GAMMA = gamma
        # decay for optimizer
        self.DECAY = decay
        # replace the target net after n iterations
        self.replace_target_iter = replace_target_iter
        # memory size
        self.memory_size = memory_size
        # batch size
        self.batch_size = batch_size
        # maximum number of episodes training
        self.train_episodes = train_episodes
        
        # create deep copy of training data
        self.data = training_data
        self.backup_data = copy.deepcopy(training_data)
        
        '___Sourcing Parameters___'
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
        
        # number of products
        self.number_products = 0
        
        # quantities of the products
        self.product_quantities = []
        for _ in range(self.max_products):
            self.product_quantities.append(0.0)
            
        # number of sources
        self.number_sources = 0
        
        # distance customer-source
        self.distances = []
        
        # stock situation
        stock_vector_template = []
        for _ in range(self.max_products):
            stock_vector_template.append(0.0)
        self.inventory_quantities = []
        for _ in range(self.max_sources):
            stock_vector_template_copy = copy.deepcopy(stock_vector_template)
            self.inventory_quantities.append(stock_vector_template_copy)
            
        # delivery
        delivery_vector_template = []
        for _ in range(self.max_products):
            delivery_vector_template.append(0.0)
        self.delivery = []
        for _ in range(self.max_sources):
            delivery_vector_template_copy = copy.deepcopy(delivery_vector_template)
            self.delivery.append(delivery_vector_template_copy)
            
        # copy product, inventory list for calculation
        self.product_quantities_c = self.product_quantities
        self.inventory_quantities_c = self.inventory_quantities
        # persist initial values of the vectors
        self.product_quantities_init = self.product_quantities
        self.inventory_quantities_init = self.inventory_quantities
        
        '___Initialize Variables, Parameters___'
        # calculate number of features
        self.n_features = self.calculate_features()
        # generate all possible states in the defined environment
        self.generate_all_actions()
        # generate indexes for the action lists
        self.generate_all_actions_index()
        # set number of actions
        self.n_actions = len(self.ALL_ACTIONS)
        # show the print statements/training, testing process
        self.PRINT_INFO = False
        
    'calculates the number of features'
    def calculate_features(self):
        num_of_features = self.max_products + self.max_sources + 2*(self.max_products*self.max_sources)
        return num_of_features
    
    'switches to the next data set for training process'
    def switch_data_training(self):
        self.ACTIONS = []
        self.ACTIONS_INDEX = []
        self.number_products = copy.deepcopy(self.backup_data.number_products)
        self.product_quantities = copy.deepcopy(self.backup_data.product_quantities)
        self.number_sources = copy.deepcopy(self.backup_data.number_sources)
        self.distances = copy.deepcopy(self.backup_data.distances)
        self.inventory_quantities = copy.deepcopy(self.backup_data.inventory_quantities)
        self.delivery = copy.deepcopy(self.backup_data.delivery)
        self.product_quantities_c = self.product_quantities
        self.inventory_quantities_c = self.inventory_quantities
        self.product_quantities_init = self.product_quantities
        self.inventory_quantities_init = self.inventory_quantities
        self.generate_actions()
        self.generate_actions_index()
    
    'Generate actions for the current episode'
    def generate_actions(self):
        for i in range(self.number_sources):
            for j in range(self.number_products):
                action_name = str(j)+","+str(i)
                self.ACTIONS.append(action_name)
        
    'Generate all possible actions for the environment according to the set parameters'    
    def generate_all_actions(self):
        for i in range(self.max_sources):
            for j in range(self.max_products):
                action_name = str(j)+","+str(i)
                self.ALL_ACTIONS.append(action_name)
              
    'Generate the index for all the possible actions'  
    def generate_all_actions_index(self):
        for i in range(len(self.ALL_ACTIONS)):
            self.ALL_ACTIONS_INDEX.append([i, self.ALL_ACTIONS[i]])
        
    'Generate the index for the current episodes actions'
    def generate_actions_index(self):
        for i in range(len(self.ACTIONS)):
            temp = self.ACTIONS[i]
            index = -1
            for j in range(len(self.ALL_ACTIONS_INDEX)):
                temp2 = self.ALL_ACTIONS_INDEX[j][1]
                if temp2 == temp:
                    index = self.ALL_ACTIONS_INDEX[j][0]
            self.ACTIONS_INDEX.append([index, temp])
        
    'Calculate the number of possible actions in the environment according to the set parameters'
    def calculate_max_actions(self):
        max_actions = self.max_products * self.max_sources
        return max_actions
             
    'Check how many sources are used to satisfy the current episodes order'
    def num_used_sources(self):
        num_used_sources = 0
        for i in range(self.max_sources):
            products_per_source = 0
            for j in range(self.max_products):
                # denormalize value
                n_delivery = self.denormalize_delivery(self.delivery[i][j])
                products_per_source += n_delivery
            if(products_per_source > 0): num_used_sources += 1
        return num_used_sources
    
    'returns a list with ids of the used sources'
    def get_used_source_ids(self):
        source_ids = []
        for i in range(self.max_sources):
            products_num = 0
            for j in range(self.max_products):
                # denormalize value
                n_delivery = self.denormalize_delivery(self.delivery[i][j])
                products_num = products_num + n_delivery
            if(products_num > 0):
                source_ids.append(i)    
        return source_ids
    
    'Check if the current episodes delivery is complete, order is satisfied'
    def check_delivery_status(self):
        to_deliver = 0
        for i in range(len(self.product_quantities_c)):
            # denormalize value
            n_pquantity = self.denormalize_pquantity(self.product_quantities_c[i])
            to_deliver += n_pquantity
        return to_deliver
    
    'Build the neural network'
    def build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2
                
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            'Trying different optimizers'
            #self._train_op = tf.train.RMSPropOptimizer(self.ALPHA).minimize(self.loss)
            self._train_op = tf.train.AdamOptimizer(self.ALPHA, self.DECAY).minimize(self.loss)
            #self._train_op = tf.train.AdadeltaOptimizer(self.ALPHA).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    'Store a state transition in the memory'
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s.observation, [a, r], s_.observation))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    'Checks if an item is out of stock'
    def item_out_of_stock(self, action):
        # check if there is still stock for the chosen product
        action_name_list = action.split(',')
        action_product = int(action_name_list[0])
        out_of_stock = True
        for i in range(len(self.inventory_quantities)):
            stock = self.inventory_quantities[i][action_product]
            stock = self.denormalize_inventory(stock)
            if(stock > 0):
                out_of_stock = False
                break
        return out_of_stock

    'Check if the current action chosen by the agent is a valid choice'
    'And check if the sourcing can be completed further or fulfillment is not possible'
    def validate_action(self, action):
        valid_action = True
        out_of_stock = None
        action_name_list = action.split(',')
        action_product = int(action_name_list[0])
        # check if the chosen product still needs more quantity
        # denormalize value
        n_product_quantities = self.denormalize_pquantity(self.product_quantities[action_product])
        if(n_product_quantities == 0.0):
            valid_action = False
        if(valid_action == True):
            out_of_stock = self.item_out_of_stock(action)
        return valid_action, out_of_stock
    
    'Calculate epsilon for the e-greedy strategy'
    def calculate_epsilon(self, episode, MAX_EPISODES):
        self.EPSILON = 1.0
        self.EPSILON = self.EPSILON - (episode / MAX_EPISODES)
    
    'Checks if all action values are zero, to determine if knowledge exists, or we need to choose random'
    def check_action_values(self, S):
        S = S.observation
        S_np = np.array(S)
        S = S_np[np.newaxis, :]
        action_values = self.sess.run(self.q_eval, feed_dict={self.s: S})
        all_zero = True
        for i in range(len(action_values)):
            if(action_values[0][i]>0): 
                all_zero = False
                break
        return all_zero
    
    'Choose the next action'
    def choose_action(self, S):
        # choose max q action
        if(np.random.uniform() >= self.EPSILON and self.check_action_values(S) == False): 
            action = self.choose_action_maxq(S)
        # choose random
        else:
            temp = []
            for i in range(len(self.ACTIONS_INDEX)):
                temp.append(self.ACTIONS_INDEX[i][0])
            action = np.random.choice(temp)
        return action
    
    'Get key of list item'
    def getListKey(self, list_item):
        # sort by q value
        return list_item[2]
    
    'Choose maximum q value action and validate it'
    def choose_action_maxq(self, S):
        # choose an action according to the max q value
        action_names = []
        S = S.observation
        S_np = np.array(S)
        S = S_np[np.newaxis, :]
        action_values = self.sess.run(self.q_eval, feed_dict={self.s: S})
        for i in range(len(self.ACTIONS_INDEX)):
            temp = self.ACTIONS_INDEX[i][0]
            temp2 = action_values[0][temp]
            action_names.append([self.ACTIONS_INDEX[i][0], self.ACTIONS_INDEX[i][1], temp2])
        action_names.sort(key=self.getListKey, reverse=True)
        #check if the actions is valid, if not choose the next best one
        for j in range(len(action_names)):
            action = action_names[j][1]
            action_id = action_names[j][0]
            valid_action, out_of_stock = self.validate_action(action)
            if(valid_action == True): 
                break
        return action_id
    
    'Add an item to the current episodes delivery list'
    def add_item_to_delivery(self, action):
        action_name_list = action.split(',')
        action_product = int(action_name_list[0])
        action_source = int(action_name_list[1])
        # remove quantity from inventory after taking the action
        self.inventory_quantities_c[action_source][action_product] -= 0.1
        self.inventory_quantities_c[action_source][action_product] = round(self.inventory_quantities_c[action_source][action_product], 1)
        # remove product item from order list
        self.product_quantities_c[action_product] -= 0.1
        self.product_quantities_c[action_product] = round(self.product_quantities_c[action_product], 1)
        # add product item to the delivery list
        self.delivery[action_source][action_product] += 0.1
        self.delivery[action_source][action_product] = round(self.delivery[action_source][action_product], 1)
        
    'Remove a product item from a source stock'
    def remove_item_from_inventory(self, action):
        action_name_list = action.split(',')
        action_product = int(action_name_list[0])
        action_source = int(action_name_list[1])
        # remove item from stock from the source where it was taken
        self.inventory_quantities[action_source][action_product] = self.inventory_quantities_c[action_source][action_product]

    'Calculate the immediate reward'
    def reward_function(self):
        A1 = 4
        A2 = 2
        A3 = 1
        
        # number of used sources
        used_sources = self.num_used_sources()
        R1 = 0 - (math.pow(2, (used_sources-1)) - 1)
        if(R1 == 0): R1 = 1
        
        # distance from customer to source
        R2 = 0
        used_source_ids = self.get_used_source_ids()
        for i in range(used_sources):
            temp_id = used_source_ids[i]
            d = self.distances[temp_id]
            #n_d = self.normalize_distance(d)
            'until 1250km +reward'
            r = 1 - (d / 0.25)
            R2 = R2 + r
        
        # remainder of stock
        R3 = 0
        stock_situation = []
        for i in range(self.max_sources):
            for j in range(self.max_products):
                # denormalize value
                n_delivery = self.denormalize_delivery(self.delivery[i][j])
                if(n_delivery > 0):
                    # denormalize value
                    n_inventory_stock = self.denormalize_inventory(self.inventory_quantities_init[i][j])
                    n_delivery = self.denormalize_delivery(self.delivery[i][j])
                    stock = n_inventory_stock - n_delivery
                    stock_situation.append(stock)
        
        for i in range(len(stock_situation)):
            r = (math.pow(2, (stock_situation[i]-1)) - 1)
            if(r < 0): r = r * 2
            if(r > 0): r = r / 4
            if(r > 2): r = 2
            R3 = R3 + r
        R3 = R3 / len(stock_situation)
        
        # combined reward
        R = A1 * R1 + A2 * R2 + A3 * R3
        
        'give additional reward when sourcing is done'
        #if(self.check_delivery_status()==0): R = R * 2    
                   
        return R
    
    'Get agent environment feedback, return new state and reward'
    def get_env_feedback(self, A):
        done = False
        # add the item to the delivery
        self.add_item_to_delivery(A)
        # calculate reward for the chosen action
        R = self.reward_function()
        # remove item from inventory where it was taken
        self.remove_item_from_inventory(A)
        # check if sourcing is done
        if(self.check_delivery_status() == 0):
            # if sourcing is finishes, return state done
            done = True
        else:
            done = False
            
        # construct next state object
        o = []
        d = []
        e = []
        s = []
        for i in range(len(self.product_quantities)):
            o.append(self.product_quantities[i])
        for i in range(self.max_sources):
            for j in range(self.max_products):
                d.append(self.delivery[i][j])
        for i in range(len(self.distances)):
            e.append(self.distances[i])
        for i in range(self.max_sources):
            for j in range(self.max_products):
                s.append(self.inventory_quantities[i][j])
        S_ = State(o, d, e, s)
        
        return S_, R, done
    
    'learning step for the NN'
    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })
        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        q_target[batch_index, eval_act_index] = reward + self.GAMMA * np.max(q_next, axis=1)
        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)
        'Epsilon is not increased atm, also should it be done here?'
        #self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    'Reinforcement Learning - training the NN'
    def train(self):
        # total learning step counter
        self.learn_step_counter = 0
        # initialize zero memory [s,a,r,s_]
        self.memory = np.zeros((self.memory_size, self.n_features*2+2))
        'multilayer perceptron'
        # build the NN
        self.build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    
        '___Variables for plotting___'
        self.cost_his = []
        self.rewards = []
        step = 0
        
        '___Training loop, train the NN for n episodes, with n orders___'
        for current_episode in range(self.train_episodes):
            
            self.switch_data_training()
            
            # print order information for the current_episode
            if(self.PRINT_INFO == True):
                print("number of products: ", self.number_products)
                print("product quantities: ", self.product_quantities)
                print("number of sources: ", self.number_sources)
                print("distances: ", self.distances)
                print("inventory quantities: ", self.inventory_quantities)
                
            reward = 0
            is_done = False
            
            # set the initial state for the current episode
            o = []
            d = []
            e = []
            s = []
            for i in range(len(self.product_quantities)):
                o.append(self.product_quantities[i])
            for i in range(self.max_sources):
                for j in range(self.max_products):
                    d.append(self.delivery[i][j])
            for i in range(len(self.distances)):
                e.append(self.distances[i])
            for i in range(self.max_sources):
                for j in range(self.max_products):
                    s.append(self.inventory_quantities[i][j])
            S = State(o, d, e, s)
            
            '___Sourcing loop, continue until the sourcing for an order is done___'
            while not is_done:
                action_valid = False
                out_of_stock = None
                
                # loop until a valid action is chosen
                while(action_valid == False):
                    
                    # agent chooses an action
                    A = self.choose_action(S)
                    
                    # get the action name of A
                    A_name = ""
                    for i in range(len(self.ACTIONS_INDEX)):
                        if(self.ACTIONS_INDEX[i][0] == A):
                            A_name = self.ACTIONS_INDEX[i][1]

                    # check if action is in Action set of current episode
                    if(A_name in self.ACTIONS):
                        # validate the chosen action
                        action_valid, out_of_stock = self.validate_action(A_name)

                # retrieve next state and the reward for the chosen action
                S_, R, done = self.get_env_feedback(A_name)
                
                # if out of stock, delivery can not be completed, terminate sourcing
                if(out_of_stock == True): done = True
                
                self.store_transition(S, A, R, S_)
                
                'the NN is updated every step atm, needs to be evaluated when it should be updated'
                # learn each step
                #if (step > 200) and (step % 5 == 0):
                self.learn()
                
                reward += R
                
                if(self.PRINT_INFO == True): 
                    print("Episode: "+str(current_episode)+", Action: "+str(A)+", Reward: "+str(R))
                    print("Delivery: "+str(self.delivery))
                    print("Out of Stock: "+str(out_of_stock))
                    
                # go to the next state
                S = S_
                # check if the sourcing is finished
                if done == True:
                    break
                step += 1

            # save reward of last episode
            self.rewards.append(reward)
        return self.rewards
    
    
    'denormalizes the product quantity'
    def denormalize_pquantity(self, n_prod_quantity):
        prod_quantity = n_prod_quantity * 10
        return prod_quantity
    
    'denormalizes the inventory stock'
    def denormalize_inventory(self, n_quantity):
        quantity = n_quantity + 1
        quantity = quantity * 10
        return quantity
    
    'denormalizes the inventory stock'
    def denormalize_delivery(self, n_delivery):
        delivery = n_delivery * 10
        return delivery