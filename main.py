import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import invwishart

import random

from mean_general import Node
from model import Model
from model import Phi # for testing purposes


def test_build(gram):
    model = Model(gram)
    traverse(model.root)
    return model

def traverse(node, visited=None):
    if visited is None:
        visited = set()

    if node in visited:
        return
    
    visited.add(node)

    # tests for build
    check_node(node)

    # tests for true mean
    # check_true_mean(node)
    
    # tests for prior mean
    # check_prior_mean(node)

    # dont call for top level
    if len(node.gram) != 1:
        traverse(node.right_par)
        traverse(node.left_par)

def check_true_mean(node):
    # test 1 is mean the correct size
    print("size p: " + str(node.get_p()))
    print("dimensions of true_mean: " + str(node.get_true_mean().shape))
    # test 2 making sure mean is not empty
    print("True Mean for " + node.gram)
    print(node.get_true_mean())

def check_prior_mean(node):
    # test 1 is mean the correct size
    print("size p: " + str(node.get_p()))
    print("dimensions of prior_mean: " + str(node.get_prior_mean().shape))
    # test 2 making sure mean is not empty
    print("Prior Mean for " + node.gram)
    print(node.get_prior_mean())

def check_node(node):
    print("Node: " + node.gram + " Node Left Parent: " + str(node.left_par)
              + " Node Right Parent: " + str(node.right_par))
    
def find_count(gram):
    return (len(gram)*(len(gram)+1))//2

def main():
    gram_3 = "THE"

    model_3 = test_build(gram_3)
    # model_3.inference(500)
    # print(model_3.get_size())

    # print(find_count(gram_3))

    # gram_ex = "EXODUS"
    # model_ex = test_build(gram_ex)
    # model_ex.inference(500)
    # print(model_ex.y.shape)
    # print(model_ex.get_size())

    # print(find_count(gram_ex))

    # phi = model_ex.get_phi()
    # for p in phi:
    #     print(p)
    #print(phi[5])

    #model_4 = test_build("THER")

    #model_ex = test_build("EXODUS")
    
    return 0
main()