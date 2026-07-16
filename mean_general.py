import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import invwishart

import random

'''
trying generalization
'''
class Node:

    mu_omega = np.array([0.3]).reshape(1,1)
    mu_delta = np.array([0.5]).reshape(1,1)
    phi_delta = np.array([
        [0],
        [1],
        [0]
    ]).reshape(3,1)

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

    # prior and true mean methods

    def update_prior_mean(self, phi_left, phi_right):
        # check edge case (level 1)
        if self.p == 1:
            eta = Node.mu_omega.copy()
        else:
            eta = (phi_left @ self.left_parent().get_est_mean() + phi_right 
                        @ self.right_parent().get_est_mean())
            # checking last edge case (level 2 needs delta as well)
            if self.p == 3:
                eta += Node.phi_delta @ Node.mu_delta
        # self.prior = self.sample_MVN(eta, self.prior_var)
        # prior should just be eta
        self.prior_mean = eta.reshape(self.p, 1)
        # check is est_mean has been set. if not,
        # est_mean = N(self. prior, self.prior_var)
        if self.est_mean is None:
            self.est_mean = self.sample_MVN(self.prior_mean, self.prior_var)

    def create_true_mean(self, phi_left, phi_right):
        # check edge case (level 1)
        if self.p == 1:
            self.true_mean = self.sample_MVN(Node.mu_omega, self.true_var)
        else:
            eta = (phi_left @ self.left_parent().get_true_mean() + phi_right 
                        @ self.right_parent().get_true_mean()).reshape(self.p, 1)
            # checking last edge case (level 2 needs delta as well)
            if self.p == 3:
                eta += Node.phi_delta @ Node.mu_delta
            # true_mean should be N(eta, true_var)
            self.true_mean = self.sample_MVN(eta, self.true_var)
        
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
    
    def sample_MVN(self, mean, cov):
        mean_mod = mean.flatten()
        return np.random.multivariate_normal(mean_mod, cov).reshape(self.p, 1)
    
    # getters
    def get_gram(self):
        return self.gram
    
    def get_p(self):
        return self.p
    
    def right_parent(self):
        return self.right_par
    
    def left_parent(self):
        return self.left_par
    
    def get_est_mean(self):
        return self.est_mean
    
    def get_true_mean(self):
        return self.true_mean
    
    def get_prior_mean(self):
        return self.prior_mean
    
    # returns ALL the children
    def get_children(self):
        return self.left_children | self.right_children
    
    def get_left_children(self):
        return self.left_children
    
    def get_right_children(self):
        return self.right_children
    
    def get_true_var(self):
        return self.true_var
    
    def __str__(self):
        return self.gram