import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import invwishart

import random

# from model import Phi

'''
trying generalization
'''
class Node:

    eta_omega = np.array([0.3])
    sig_omega = invwishart.rvs(
                df=(3), # self.p=1 + 2
                scale= np.eye(1) # self.p=1
            ).reshape(1, 1) #self.p=1
    mu_omega = np.random.multivariate_normal(eta_omega, sig_omega).reshape(1, 1)
    sig_delta = invwishart.rvs(
                df=(3), # self.p=1 + 2
                scale= np.eye(1) # self.p=1
            ).reshape(1, 1) #self.p =1
    eta_delta = np.array([0.5])
    phi_delta = np.array([
        [0],
        [1],
        [0]
    ]).reshape(3,1)
    mu_delta = np.random.multivariate_normal(eta_delta, sig_delta).reshape(1, 1)

    M = [500, 100, 10, 2, 0]
    
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
        self.prior_var_inv = np.linalg.inv(self.prior_var)
        self.post_var = None

        self.prior_var_mod_left = None
        self.prior_var_mod_right = None

        # mean variables
        self.true_mean = None

        # eta(s)
        self.prior_mean = None
        self.post_mean = None

        # what will be compared to true_mean
        self.est_mean = None
        self.mean_copies = []

        # cycle boolean
        self.has_run = False
        self.has_run_prior = False

        # data
        self.y = None
        self.m = Node.M[random.randint(0, 4)]

    def check_post_mean(self, phi_collection):
        for c in self.get_children():
            if not c.get_has_run():
                c.posterior_mean(phi_collection)

    # updated est mean
    def posterior_mean(self, phi_collection):
        self.check_post_mean(phi_collection)
        if self.has_run:
            return
        if self.y is not None:
            self.posterior_data()
            return
        # print(self.gram)
        phi = phi_collection[len(self.gram)]
        phi_left = phi.phi_left()
        phi_right = phi.phi_right()
        # first sample should be from prior
        # second we can calculate the variance but then it shouldn't change
        if len(self.mean_copies) < 2:
            self.posterior_mean_first_run(phi_left, phi_right)
            return
        # post variance first
        # post_V = prior_V.copy() # inverse to make into post V

        eta = self.prior_mean.copy()
        post_eta = self.prior_var_inv @ eta

        for c in self.get_left_children():
            # variance
            child_V = c.get_prior_var_mod_right()
            # post_V += phi_right.T @ child_V @ phi_right

            # eta
            sib_mean = c.left_parent().get_est_mean()
            child_mean = c.get_est_mean()
            # post_eta += child_V @ (child_mean - 
            #                                     phi_left @ sib_mean)
            post_eta += child_V @ (child_mean - 
                                                phi_left @ sib_mean)
            
        for c in self.get_right_children():
            # variance
            child_V = c.get_prior_var_mod_left()
            # post_V += phi_left.T @ child_V @ phi_left

            # eta
            sib_mean = c.right_parent().get_est_mean()
            child_mean = c.get_est_mean()
            post_eta += child_V @ (child_mean - 
                                                phi_right @ sib_mean)

        

        # self.post_var = np.linalg.inv(post_V)
        self.post_mean = self.post_var @ post_eta

        # self.prior_var = self.post_var.copy()

        self.est_mean = self.sample_MVN(self.post_mean, self.post_var)
        self.mean_copies.append(self.est_mean.copy())

        self.has_run = True
        self.has_run_prior = False
        # self.run_sample_mean()

    def posterior_mean_first_run(self, phi_left, phi_right):
        # post variance first
        post_V = self.prior_var_inv.copy() # inverse to make into post V
        # self.prior_var_mod_left = phi_left @ self.prior_var_inv
        # self.prior_var_mod_right = phi_right @ self.prior_var_inv

        eta = self.prior_mean.copy()
        post_eta = self.prior_var_inv @ eta

        for c in self.get_left_children():
            # variance
            child_V = c.get_prior_var_mod_right()
            post_V += child_V @ phi_right

            # eta
            sib_mean = c.left_parent().get_est_mean()
            child_mean = c.get_est_mean()
            post_eta += child_V @ (child_mean - 
                                                phi_left @ sib_mean)
            
        for c in self.get_right_children():
            # variance
            child_V = c.get_prior_var_mod_left()
            post_V += child_V @ phi_left

            # eta
            sib_mean = c.right_parent().get_est_mean()
            child_mean = c.get_est_mean()
            post_eta += child_V @ (child_mean - 
                                                phi_right @ sib_mean)

        

        self.post_var = np.linalg.inv(post_V)
        self.post_mean = self.post_var @ post_eta

        self.est_mean = self.sample_MVN(self.post_mean, self.post_var)
        self.mean_copies.append(self.est_mean.copy())

        self.has_run = True
        self.has_run_prior = False

    def data(self):
        self.y = np.random.multivariate_normal(
            self.get_true_mean().flatten(),
            self.get_true_var(),
            size=self.m
        )
            

    def posterior_data(self):
        # if we have already calculated posterior, just need to sample again
        if len(self.mean_copies) > 2:
            self.est_mean = self.sample_MVN(self.post_mean, self.post_var)
            self.mean_copies.append(self.est_mean.copy())
            self.has_run = True
            self.has_run_prior = False
            return
        # post variance wow
        true_sig = np.linalg.inv(self.true_var)
        n = self.y.shape[0]
        self.post_var = np.linalg.inv(self.prior_var_inv + n* true_sig)

        # post eta
        y_bar = np.mean(self.y, axis=0).reshape(self.p, 1)
        eta = self.prior_mean
        # print(self.gram)
        self.post_mean = self.post_var @ (self.prior_var_inv@eta + n*true_sig @ y_bar)

        # self.prior_var = self.post_var.copy()

        self.est_mean = self.sample_MVN(self.post_mean, self.post_var)
        self.mean_copies.append(self.est_mean.copy())

        self.has_run = True
        self.has_run_prior = False
        #self.prior_mean = None
        # self.run_sample_mean()

    def get_final_mean_est(self):
        num_samples = len(self.mean_copies)

        burn = int(num_samples*0.1)

        mu_post = np.array(self.mean_copies[burn:])
        mu_est = mu_post.mean(axis=0)

        self.est_mean = np.array(mu_est).reshape(self.p, 1)
        # self.print_results()

    
    def run_sample_mean(self):
        num_samples = 1000

        mu_samples = []

        # sample a bunch
        for i in range(num_samples):

            mu = self.sample_MVN(self.post_mean, self.post_var)

            mu_samples.append(mu.copy())

        burn = int(num_samples*0.1)

        mu_post = np.array(mu_samples[burn:])
        mu_est = mu_post.mean(axis=0)

        self.est_mean = np.array(mu_est).reshape(self.p, 1)

    def print_results(self):
        string = "*******************\n"
        print("*******************")
        print("For gram "+ self.gram)
        string += "For gram "+ self.gram + "\n"
        print("True Mean: ")
        print(self.true_mean)
        print("Estimated Mean: ")
        print(self.est_mean)
        print("Initial Mean: ")
        print(self.mean_copies[0])
        print("Difference in true and est mean")
        string += "Difference in true and est mean\n\n"
        for i in range(self.true_mean.shape[0]):
            print(self.true_mean[i][0] - self.est_mean[i][0])
            string += str(self.true_mean[i][0] - self.est_mean[i][0]) +"\n\n"
        # self.write_to_file(string)
        # writing to save file
        

    def write_to_file(self, string):
        with open("../track_mean_est.md", "a") as file:
            file.writelines(string)

    # prior and true mean methods

    def update_prior_mean(self, phi_collection):
        if self.check_parents():
            self.left_par.update_prior_mean(phi_collection)
            self.right_par.update_prior_mean(phi_collection)
        if self.has_run_prior:
            # if self.check_parents():
            #     self.left_par.update_prior_mean(phi_collection)
            #     self.right_par.update_prior_mean(phi_collection)
            return
        self.has_run = False
        # check edge case (level 1)
        if self.p == 1:
            eta = Node.mu_omega
        else:
            phi = phi_collection[len(self.gram)-1]
            phi_left = phi.phi_left()
            phi_right = phi.phi_right()
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
        self.has_run_prior = True
        if self.est_mean is None:
            self.est_mean = self.sample_MVN(self.prior_mean, self.prior_var)
        

    def create_true_mean(self, phi_collection):
        if self.check_parents():
            self.left_par.create_true_mean(phi_collection)
            self.right_par.create_true_mean(phi_collection)
        if self.true_mean is not None:
            return
        # check edge case (level 1)
        if self.p == 1:
            self.true_mean = self.sample_MVN(Node.mu_omega, self.true_var)
            self.prior_var_mod_right = self.prior_var_inv.copy()
            self.prior_var_mod_left = self.prior_var_inv.copy()
        else:
            phi = phi_collection[len(self.gram)-1]
            phi_left = phi.phi_left()
            phi_right = phi.phi_right()
            eta = (phi_left @ self.left_parent().get_true_mean() + phi_right 
                        @ self.right_parent().get_true_mean()).reshape(self.p, 1)
            self.prior_var_mod_right = phi_right.T @ self.prior_var_inv
            self.prior_var_mod_left = phi_left.T @ self.prior_var_inv
            # checking last edge case (level 2 needs delta as well)
            if self.p == 3:
                eta[1][0] = Node.mu_delta[0][0]
            # true_mean should be N(eta, true_var)
            # self.true_mean = eta.reshape(self.p, 1)
            self.true_mean = self.sample_MVN(eta, self.true_var)

        # if len(phi_collection) != len(self.gram):
        #     phi = phi_collection[len(self.gram)-1]
        #     phi_left = phi.phi_left()
        #     phi_right = phi.phi_right()
        #     self.prior_var_mod_right = phi_right @ self.prior_var_inv
        #     self.prior_var_mod_left = phi_left @ self.prior_var_inv

        if self.has_no_children():
            self.data()
            
            
        
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
    
    def has_no_children(self):
        return len(self.left_children) == 0 and len(self.right_children) == 0
    
    def get_left_children(self):
        return self.left_children
    
    def get_right_children(self):
        return self.right_children
    
    def get_true_var(self):
        return self.true_var
    
    def get_prior_var(self):
        return self.prior_var

    def get_prior_var_inv(self):
        return self.prior_var_inv

    def get_prior_var_mod_left(self):
        return self.prior_var_mod_left

    def get_prior_var_mod_right(self):
        return self.prior_var_mod_right
    
    def get_post_var(self):
        return self.post_var
    
    def get_has_run(self):
        return self.has_run
    
    def get_has_run_prior(self):
        return self.has_run_prior

    def get_m(self):
        return self.m
    
    def __str__(self):
        return self.gram



class Phi:

    def __init__(self, level):
        self.left = None
        self.right = None
        self.level = level
        # number of rows = parameters
        self.p = 2*level - 1
        # number of cols (parameters of previous levels)
        if self.p==1:
            self.k = 1
        else:
            self.k = 2*(level-1) - 1

        # build
        self.build()
        
    def build(self):
        self.right = np.zeros((self.p, self.k)) # dimensions (parameters of level mean, par of prev level mean)
        self.left = np.zeros((self.p, self.k))
        # level 1 is only 1x1
        if self.level == 1:
            self.left[0][0] = 1
            self.right[0][0] = 1
            return
        # level 2 is different because of delta
        if self.level == 2:
            self.left[0][0] = 1
            self.right[2][0] = 1
            return
        # for all other levels

        # starting with left first two diagonals are 1
        self.left[0][0] = 1
        self.left[1][1] = 1
        # rest are 0.5
        i = 2
        while i < self.k:
            self.left[i][i] = 0.5
            i+=1

        # right
        i = 2
        j = 0
        # want to stop 2 steps before p
        while i < (self.p - 2):
            self.right[i][j] = 0.5
            i += 1
            j += 1
        # setting values at 1
        self.right[i][j] = 1
        i+= 1
        j += 1
        self.right[i][j] = 1
        return
    
    # getters for each phi
    def phi_left(self):
        return self.left
    
    def phi_right(self):
        return self.right
    
    def __str__(self):
        string = "Left Phi at level " + str(self.level) + "\n" + str(self.left) + "\n"
        string += "Right Phi at level " + str(self.level) + "\n" + str(self.right)
        return string