import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import invwishart

import random

import string

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from mean_general import Node
from model import Model
import math

from itertools import product
import graphs as gr


def test_build(gram, n):
    model = Model(gram)
    model.inference()
    print_nodes(model)
    return model

def build(gram):
    model = Model(gram)
    # ch = string.printable
    ch = string.ascii_uppercase
    print("Number of characters added: " + str(len(ch)))
    for l in list(ch):
        model.add_gram(gram[:-1] + l)
        model.add_gram(l + gram[:-1])
        model.add_gram(gram[1:] + l)
        model.add_gram(l + gram[1:])
    # # print(model.nodes["ITH"].y)
    model.inference()
    # model.nodes["TH"].print_results()
    # model.nodes["HE"].print_results()
    # for gram in model.nodes:
    #     model.nodes[gram].print_results()
    return model

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

def print_nodes(model):
    for gram in model.nodes:
        model.nodes[gram].print_results()




def get_same_list(all_grams):
    grams = []
    no_tri_grams = [g for g in all_grams if len(g) != 3]
    unique_numbers = random.sample(range(0, len(no_tri_grams)), 10)
    for i in range(len(unique_numbers)):
        grams.append(no_tri_grams[unique_numbers[i]])
    return grams

def build_full():
    gram = "THE"
    model = Model(gram)
    # ch = string.printable
    ch = string.ascii_uppercase
    combinations = [''.join(item) for item in product(ch, repeat=3)]
    combinations = combinations[1000:]
    print("Number of characters added: " + str(len(combinations)))
    for c in combinations:
        model.add_gram(c)
    # # print(model.nodes["ITH"].y)
    model.inference()
    gr.make_one_learning_plot(model)
    return model

def main():
    gram_3 = "THE"
    model = build(gram_3)
    # for g in model.nodes:
    #     model.nodes[g].print_results()

    
    # Error-based redesign (highlights the effectiveness of learning):
    gr.make_learning_error_plot(model)
    gr.make_error_vs_m_plot(model)
    gr.make_trigram_learning_plot(model)
    gr.make_posterior_predictive_check(model)

    # Old level-based versions (kept for reference):
    # gr.make_one_learning_plot(model)
    # gr.make_M_and_predictive(model)


    # build_full()
    
    return 0
main()