import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import invwishart

import random

import time

from mean_general import Node
from mean_general import Phi


class Model:

    def __init__(self, root):
        self.root = Node(root)
        self.nodes = {}
        self.nodes[self.root.get_gram()] = self.root

        self.roots = set()
        self.roots.add(self.root)

        self.depth = len(self.root.get_gram())
        self.level1 = set()

        self.build(self.root)
        # create data for lower nodes
      

        # phi stuff
        self.phi_collection = [] # go from level 1 to what is needed
     

    def update_posterior(self):
        # need to go from down up
        return

    def inference(self):
        start_time = time.perf_counter()
        self.build_phi()
        self.create_true_mean()
        self.run_marginal_prior()
        self.update_prior_mean()   
        # step 1: make data
        # self.data(n)

        # step 2: last level posterior
        # just for checking if it works
        # print("n = "+str(self.n))
        
        
        self.gibbs()
        end_time = time.perf_counter()
        execution_time = end_time - start_time

        print(f"Inference Method Executed in: {execution_time:.4f} seconds")
      

    def gibbs(self):
        num_samples = 1000
        print("Number of Gibbs samples: "+ str(num_samples))

        # sample a bunch
        for i in range(num_samples):
            # self.root.posterior_data(self.y)
            
            # call something that goes down up
            # update our posteriors
            self.update_posterior_mean()

            # call something that goes up down
            # update priors
            self.update_prior_mean()

            #mu = self.sample_MVN(self.post_mean, self.post_var)

            # for gram in self.nodes:
            #     self.nodes[gram].has_run = False

        for gram in self.nodes:
            self.nodes[gram].get_final_mean_est()

   
    def run_marginal_prior(self):
        num = 1000
        for i in range(num):
            for node in self.roots:
                node.marginal_prior_update(self.phi_collection)


    def update_posterior_mean(self):
        # need to go up the tree once and call posterior for each node
        # for c in node.get_children():
        #     if not c.get_has_run():
        #         self.update_posterior_mean(c)
        for node in self.level1:
            node.posterior_mean(self.phi_collection)


        
        return
        

    def build(self, node):
        gram = node.get_gram()
        # has no more parent nodes
        if len(gram) <= 1:
            if len(gram) == 1:
                self.level1.add(node)
            return
        gram_left = gram[:-1] # removing last char

        gram_right = gram[1:] # removing first character
        self.connect_left(gram_left, node)
        self.connect_right(gram_right, node)

        return
    
    def add_gram(self, gram):
        new_node = Node(gram)
        if gram in self.nodes:
            return
        self.nodes[gram] = new_node

        if len(gram) > self.depth:
            self.depth = len(gram)

        self.roots.add(new_node)

        self.build(new_node)
        # redo true and prior mean
        # add more phi's if need be
    
    def connect_right(self, gram_right, child):
        if gram_right in self.nodes:
            par_right = self.nodes[gram_right]
            # if so add child to existing left parent and set current node left parents
            # self.connect_right(par_right, node)
            child.set_right_parent(par_right)
            par_right.add_left(child)
        else:
            # if not add node and set parent and child (node)
            par_right = Node(gram_right)
            self.nodes[gram_right] = par_right
            # self.connect_right(par_right, node)
            self.build(par_right)
            child.set_right_parent(par_right)
            par_right.add_left(child)

    def connect_left(self, gram_left, child):
        if gram_left in self.nodes:
            par_left = self.nodes[gram_left]
            # if so add child to existing left parent and set current node left parents
            # self.connect_left(gram_left, node)
            child.set_left_parent(par_left)
            par_left.add_right(child)
        else:
            # if not we add the node and set
            par_left = Node(gram_left)
            self.nodes[gram_left] = par_left
            # self.connect_left(par_left, node)
            self.build(par_left)
            child.set_left_parent(par_left)
            par_left.add_right(child)

    # build all of relevant phi's
    def build_phi(self):
        for i in range(1, self.depth+1):
            self.phi_collection.append(Phi(i))
        return
    
    def create_true_mean(self):
        # already been set

        for gram in self.nodes:
            self.nodes[gram].create_true_mean(self.phi_collection)
       

    def update_prior_mean(self):
        # if visited is None:
        #     visited = set()

        for node in self.roots:
            node.update_prior_mean(self.phi_collection)


    def get_size(self):
        return len(self.nodes)
    
    def get_phi(self):
        return self.phi_collection
    

