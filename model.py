import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import invwishart

import random

from mean_general import Node

class Model:

    def __init__(self, root):
        self.root = Node(root)
        self.nodes = {}
        self.nodes[self.root.get_gram()] = self.root
        self.y = None
        self.level1 = []

        self.build(self.root)

        # phi stuff
        self.phi_collection = [] # go from level 1 to what is needed
        # rules of indexing
        # # correct phi object is as len(gram)-1 -> level-1
        self.build_phi()        
        

    def build(self, node):
        gram = node.get_gram()
        # has no more parent nodes
        if len(gram) == 0:
            return
        if len(gram) == 1:
            self.level1.append(node)
            return
        gram_left = gram[:-1] # removing last char

        gram_right = gram[1:] # removing first character

        # check if left parents is in nodes
        if gram_left in self.nodes:
            par_left = self.nodes[gram_left]
            # if so add child to existing left parent and set current node left parents
            self.connect_left(par_left, node)
        else:
            # if not we add the node and set
            par_left = Node(gram[:-1])
            self.nodes[gram[:-1]] = par_left
            self.connect_left(par_left, node)
            self.build(par_left)
        
        # checking right parent
        if gram_right in self.nodes:
            par_right = self.nodes[gram_right]
            # if so add child to existing left parent and set current node left parents
            self.connect_right(par_right, node)
        else:
            # if not add node and set parent and child (node)
            par_right = Node(gram[1:])
            self.nodes[gram[1:]] = par_right
            self.connect_right(par_right, node)
            self.build(par_right)

        # if both parents are set we can return
        return
    
    def connect_right(self, par_right, child):
        child.set_right_parent(par_right)
        par_right.add_left(child)

    def connect_left(self, par_left, child):
        child.set_left_parent(par_left)
        par_left.add_right(child)

    # build all of relevant phi's
    def build_phi(self):
        for i in range(1, len(self.root.get_gram())+1):
            self.phi_collection.append(Phi(i))
        return
    
    def get_size(self):
        return len(self.nodes)
    
    def get_phi(self):
        return self.phi_collection
    

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