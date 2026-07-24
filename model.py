import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import invwishart

import random

import time

from mean_general import Node
from mean_general import Phi

'''
    Eventually to try:
        - make a list of first nodes and root nodes
        - probably faster iteration
'''

class Model:

    def __init__(self, root):
        self.root = Node(root)
        self.nodes = {}
        self.nodes[self.root.get_gram()] = self.root

        self.roots = set()
        self.roots.add(self.root)

        self.depth = len(self.root.get_gram())

        self.build(self.root)
        # create data for lower nodes
      

        # phi stuff
        self.phi_collection = [] # go from level 1 to what is needed
        # rules of indexing
        # # correct phi object is as len(gram)-1 -> level-1
    

        # change from list to just one object
        self.level1 = self.nodes[self.root.get_gram()[0]]
        # create true mean
        # self.create_true_mean(self.level1)
        # self.update_prior_mean(self.level1)   
        # for gram in self.nodes:
        #     node = self.nodes[gram]
        #     if node.has_no_children() and node.y is None:
        #         node.data(self.n)

    # inference yay! :D

    # def data(self, n):
    #     self.y = np.random.multivariate_normal(
    #         self.root.get_true_mean().flatten(),
    #         self.root.get_true_var(),
    #         size=n
    #     )

    def update_posterior(self):
        # need to go from down up
        return

    def inference(self):
        self.build_phi()
        self.create_true_mean(self.level1)
        self.update_prior_mean(self.level1)   
        # step 1: make data
        # self.data(n)

        # step 2: last level posterior
        # just for checking if it works
        # print("n = "+str(self.n))
        start_time = time.perf_counter()
        
        self.gibbs()
        end_time = time.perf_counter()
        execution_time = end_time - start_time

        print(f"Executed in: {execution_time:.4f} seconds")
        # for k in self.nodes:
        #     node = self.nodes[k]
        #     node.print_results()
        # self.root.get_final_mean_est()
        # self.root.posterior_data(self.y)
        # self.update_posterior(self.root)
        # self.root.run_sample_mean()
        # self.root.print_results()

    def gibbs(self):
        num_samples = 1000
        print("Number of Gibbs samples: "+ str(num_samples))

        # sample a bunch
        for i in range(num_samples):
            # self.root.posterior_data(self.y)
            
            # call something that goes down up
            # update our posteriors
            self.update_posterior_mean(self.root)

            # call something that goes up down
            # update priors
            self.update_prior_mean(self.level1)

            #mu = self.sample_MVN(self.post_mean, self.post_var)

            # for gram in self.nodes:
            #     self.nodes[gram].has_run = False

        for gram in self.nodes:
            self.nodes[gram].get_final_mean_est()

    # def update_posterior_mean_data(self, node):
    #     if node.get_has_run():
    #         return
        
    #     node.posterior_data()



    def update_posterior_mean(self, node):
        # need to go up the tree once and call posterior for each node
        # for c in node.get_children():
        #     if not c.get_has_run():
        #         self.update_posterior_mean(c)
        for gram in self.nodes:
            self.nodes[gram].posterior_mean(self.phi_collection)


        # # node has already been run
        # if node.get_has_run():
        #     if node.check_parents():
        #         self.update_posterior_mean(node.left_parent())
        #         self.update_posterior_mean(node.right_parent())
        #     return

        # # print(node.gram)

        # level = len(node.gram) +1
        # if node.has_no_children():
        #     node.posterior_mean(None, None)
        # else:
        #     phi = self.phi_collection[level-1]
        #     node.posterior_mean(phi.phi_left(), phi.phi_right())
        
        # # if node does not have parents we are at the top
        # if node.check_parents():
        #     self.update_posterior_mean(node.left_parent())
        #     self.update_posterior_mean(node.right_parent())
        
        return
        

    def build(self, node):
        gram = node.get_gram()
        # has no more parent nodes
        if len(gram) <= 1:
            return
        gram_left = gram[:-1] # removing last char

        gram_right = gram[1:] # removing first character
        self.connect_left(gram_left, node)
        self.connect_right(gram_right, node)

        # check if left parents is in nodes
        # if gram_left in self.nodes:
        #     par_left = self.nodes[gram_left]
        #     # if so add child to existing left parent and set current node left parents
        #     self.connect_left(gram_left, node)
        # else:
        #     # if not we add the node and set
        #     par_left = Node(gram[:-1])
        #     self.nodes[gram[:-1]] = par_left
        #     self.connect_left(par_left, node)
        #     self.build(par_left)
        
        # checking right parent
        # if gram_right in self.nodes:
        #     par_right = self.nodes[gram_right]
        #     # if so add child to existing left parent and set current node left parents
        #     self.connect_right(par_right, node)
        # else:
        #     # if not add node and set parent and child (node)
        #     par_right = Node(gram[1:])
        #     self.nodes[gram[1:]] = par_right
        #     self.connect_right(par_right, node)
        #     self.build(par_right)

        # if both parents are set we can return
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
    
    def create_true_mean(self, node):
        # already been set

        for gram in self.nodes:
            self.nodes[gram].create_true_mean(self.phi_collection)
        # if node.get_true_mean() is not None:
        #     return

        # # if level 1 we can skip this step
        # # ensuring all parents are visited first
        # if node.check_parents():
        #     if node.left_parent().get_true_mean() is None:
        #         self.create_true_mean(node.left_parent())
        #     if node.right_parent().get_true_mean() is None:
        #         self.create_true_mean(node.right_parent())

        # # all parents visited or 1st level node
        # if node.get_true_mean() is None:
        #     level = len(node.gram)
        #     phi_level = self.phi_collection[level-1]
        #     # passing correct phi left and right
        #     node.create_true_mean(phi_level.phi_left(),
        #                      phi_level.phi_right())

        # for child in node.get_children():
        #     self.create_true_mean(child)

    def update_prior_mean(self, node):
        # if visited is None:
        #     visited = set()

        for gram in self.nodes:
            self.nodes[gram].update_prior_mean(self.phi_collection)

        # if node.get_has_run_prior():
        #     return
        # # if level 1 we can skip this step
        # # ensuring all parents are visited first
        # if node.check_parents():
        #     if not node.left_parent().get_has_run_prior():
        #         self.update_prior_mean(node.left_parent())
        #     if not node.right_parent().get_has_run_prior():
        #         self.update_prior_mean(node.right_parent())

        # # all parents visited or 1st level node
        # if not node.get_has_run_prior():
        #     level = len(node.gram)
        #     phi_level = self.phi_collection[level-1]
        #     # passing correct phi left and right
        #     node.update_prior_mean(phi_level.phi_left(),
        #                      phi_level.phi_right())

        # for child in node.get_children():
        #     self.update_prior_mean(child)


    def get_size(self):
        return len(self.nodes)
    
    def get_phi(self):
        return self.phi_collection
    

