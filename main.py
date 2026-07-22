import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import invwishart

import random

import string

import sys

sys.setrecursionlimit(2000)

from mean_general import Node
from model import Model
from model import Phi # for testing purposes


def test_build(gram, n):
    model = Model(gram, n)
    print("Number of characters added: " + str(len(string.printable)))
    for l in list(string.printable):
        model.add_gram(gram[:-1] + l)
        model.add_gram(l + gram[:-1])
        model.add_gram(gram[1:] + l)
        # model.add_gram(l + gram[1:])
    # model.add_gram("THB")
    # model.add_gram("THA")
    # model.add_gram("THC")
    # model.add_gram("THD")
    # model.add_gram("THF")
    # model.add_gram("THG")
    # model.add_gram("THH")
    # model.add_gram("THJ")
    # model.add_gram("THL")
    # model.add_gram("THM")

    # for gram in model.nodes:
    #     check_node(model.nodes[gram])
    # traverse(model.root)
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
    # gram_3 = "THE"
    # gram_3 = "EXODUS"
    gram_3 = "THE"

    model_3 = test_build(gram_3, 500)
    # for c in model_3.nodes["TH"].right_children:
    #     print(c.gram)
    # print(model_3.nodes["HE"].left_children.gram)
    model_3.inference()
    model_3.nodes["TH"].print_results()
    model_3.nodes["HE"].print_results()
    # model_3.nodes["T"].print_results()
    # model_3.nodes["THE"].print_results()
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