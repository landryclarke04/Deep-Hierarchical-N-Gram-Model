import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import invwishart

import random

'''
trying generalization
'''
class Node:

    def __init__(self, gram):
        self.gram = gram
        # number of parameters (for size of mean and cov etc) 2n - 1
        self.p = 2*len(gram) - 1
        # parents
        self.left_par = None
        self.right_par = None
        # set of relatives
        self.left_children = set()
        self.right_children = set()
        # figure out phi situation

        # variance variables
        self.true_var = self.get_IWH_var()
        self.prior_var = self.get_IWH_var()
        self.post_var = None

        # mean variables
        self.true_mean = None

        # eta(s)
        self.prior_mean = None
        self.post_mean = None

        # what will be compared to true_mean
        self.est_mean = None

    # setters

    def set_right_parent(self, parent):
        self.right_par = parent

    def set_left_parent(self, parent):
       self.left_par = parent

    # check if both parents are set. Used in build
    def check_parents(self):
        return self.left_par is not None and self.right_par is not None

    # add to relatives
    def add_left(self, ch):
        #ch.set_right_parent(self)
        self.left_children.add(ch)

    def add_right(self, ch):
        #ch.set_left_parent(self)
        self.right_children.add(ch)


    def get_IWH_var(self): # method checked
        v = np.eye(self.p)
        return invwishart.rvs(
                df=(self.p+2),
                scale=v
            ).reshape(self.p, self.p)
    
    # getters
    def get_gram(self):
        return self.gram
    
    def get_p(self):
        return self.p
    
    def __str__(self):
        return self.gram