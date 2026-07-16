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
        self.build(self.root)
        

    def build(self, node):
        gram = node.get_gram()
        # has no more parent nodes
        if len(gram) <= 1:
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
    
    def get_size(self):
        return len(self.nodes)
    
