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

    if node in visited or len(node.gram)==1:
        return
    
    visited.add(node)

    check_node(node)
    
    traverse(node.right_par)
    traverse(node.left_par)


def check_node(node):
    print("Node: " + node.gram + " Node Left Parent: " + str(node.left_par)
              + " Node Right Parent: " + str(node.right_par))
    
def find_count(gram):
    return (len(gram)*(len(gram)+1))//2

def main():
    # gram_3 = "THE"

    # model_3 = test_build(gram_3)
    # print(model_3.get_size())

    # print(find_count(gram_3))

    gram_ex = "EXODUS"
    model_ex = test_build(gram_ex)
    # print(model_ex.get_size())

    # print(find_count(gram_ex))

    phi = model_ex.get_phi()
    # for p in phi:
    #     print(p)
    #print(phi[5])

    #model_4 = test_build("THER")

    #model_ex = test_build("EXODUS")
    




    # node = Node("THE")
    # print(node.true_var)
    # child = Node("TH")
    # print(child.true_var)
    # print(node)

    # pair = node, child
    # print(pair)
    # print(pair[1].gram)
    # child.gram = "Hello"
    # print(pair[1].gram)
    return 0
main()