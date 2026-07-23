import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import invwishart

import random

import string

import matplotlib.pyplot as plt

from mean_general import Node
from model import Model


def test_build(gram, n):
    model = Model(gram, n)
    model.inference()
    print_nodes(model)
    return model

def build(gram):
    n = 10
    model = Model(gram, n)
    ch = string.printable
    # ch = string.ascii_uppercase
    print("Number of characters added: " + str(len(ch)))
    # for l in list(ch):
    #     model.add_gram(gram[:-1] + l)
    #     model.add_gram(l + gram[:-1])
    #     model.add_gram(gram[1:] + l)
        # model.add_gram(l + gram[1:])
    # print(model.nodes["ITH"].y)
    model.inference()
    # for g in model.nodes:
    #     node = model.nodes[g]
    #     # print("clear")
    #     if node.y is not None:
    #         print(g)
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
    model.nodes["TH"].print_results()
    model.nodes["HE"].print_results()

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

def print_nodes(model):
    for gram in model.nodes:
        model.nodes[gram].print_results()

def make_plot_prior(model):
    prior_means = []
    prior_stds = []
    true_means = []
    # starting with 5
    grams = ["T", "TH", "HE", "H", "E"]
    for g in grams:
        
        node = model.nodes[g]
        i = random.randint(0, node.p-1)
        prior_means.append(node.mean_copies[0].flatten()[i])
        #prior_std_scalar = [np.sqrt(Sigma[0, 0]) for Sigma in covariances]
        prior_std = np.sqrt((node.prior_var[i][i]))
        # prior_std = 0.1 * abs(node.mean_copies[0].flatten()[i])
        # print(np.sqrt(np.diag(node.prior_var)))
        prior_stds.append(prior_std)
        true_means.append(node.get_true_mean().flatten()[i])
    # x = np.arange(1, 11)  # positions 1 through 10
    # x = np.arange(1,6)  # positions 1 through 5
    x = grams
    plt.errorbar(
        x,
        prior_means,
        # yerr=2*prior_stds,
        yerr=prior_stds,
        fmt='o',
        label='Prior Mean'
    )

    plt.scatter(
        x,
        true_means,
        color='red',
        marker='*',
        s=150,
        label='True Mean'
    )
    plt.xlabel('Case')
    plt.ylabel('Mean')
    plt.legend()
    plt.show()
    # labels = grams
    # #plt.xticks(x, labels)
    # fig, ax = plt.subplots()
    # ax.set_xticks(x)
    # ax.set_xticklabels(labels)

def make_plot_post(model):
    post_means = []
    post_stds = []
    true_means = []
    # starting with 5
    grams = ["T", "TH", "HE", "H", "E"]
    for g in grams:
        
        node = model.nodes[g]
        i = random.randint(0, node.p-1)
        post_means.append(node.get_est_mean().flatten()[i])
        #prior_std_scalar = [np.sqrt(Sigma[0, 0]) for Sigma in covariances]
        post_std = np.sqrt((node.post_var[i][i]))
        # prior_std = 0.1 * abs(node.mean_copies[0].flatten()[i])
        # print(np.sqrt(np.diag(node.prior_var)))
        post_stds.append(post_std)
        true_means.append(node.get_true_mean().flatten()[i])
    # x = np.arange(1, 11)  # positions 1 through 10
    # x = np.arange(1,6)  # positions 1 through 5
    x = grams
    plt.errorbar(
        x,
        post_means,
        # yerr=2*prior_stds,
        yerr=post_stds,
        fmt='o',
        label='Post Mean'
    )

    plt.scatter(
        x,
        true_means,
        color='red',
        marker='*',
        s=150,
        label='True Mean'
    )
    plt.xlabel('Case')
    plt.ylabel('Mean')
    plt.legend()
    plt.show()

def make_leanring_plot(model):
    fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(12, 5),
    sharey=True
    )
    prior_means = []
    prior_stds = []
    true_means = []
    post_means = []
    post_stds = []
    # starting with 5
    grams = ["T", "TH", "HE", "H", "E"]
    x = grams
    labels = grams
    for g in grams:
        
        node = model.nodes[g]
        i = random.randint(0, node.p-1)
        prior_means.append(node.mean_copies[0].flatten()[i])
        #prior_std_scalar = [np.sqrt(Sigma[0, 0]) for Sigma in covariances]
        prior_std = np.sqrt((node.prior_var[i][i]))
        # prior_std = 0.1 * abs(node.mean_copies[0].flatten()[i])
        # print(np.sqrt(np.diag(node.prior_var)))
        prior_stds.append(prior_std)
        true_means.append(node.get_true_mean().flatten()[i])
        post_means.append(node.get_est_mean().flatten()[i])
        #prior_std_scalar = [np.sqrt(Sigma[0, 0]) for Sigma in covariances]
        post_std = np.sqrt((node.post_var[i][i]))
        # prior_std = 0.1 * abs(node.mean_copies[0].flatten()[i])
        # print(np.sqrt(np.diag(node.prior_var)))
        post_stds.append(post_std)

    # Prior
    ax1.errorbar(
        x,
        prior_means,
        yerr=prior_stds,
        fmt='o',
        capsize=4,
        label='Prior'
    )

    ax1.scatter(
        x,
        true_means,
        color='red',
        marker='*',
        s=120,
        label='Truth'
    )

    ax1.set_title('Prior')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45)

    # Posterior
    ax2.errorbar(
        x,
        post_means,
        yerr=post_stds,
        fmt='o',
        capsize=4,
        label='Posterior'
    )

    ax2.scatter(
        x,
        true_means,
        color='red',
        marker='*',
        s=120,
        label='Truth'
    )

    ax2.set_title('Posterior')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45)

    plt.tight_layout()
    plt.legend()
    plt.show()

def main():
    # gram_3 = "THE"
    # gram_3 = "EXODUS"
    gram_3 = "THE"

    # model_3 = test_build(gram_3, 500)
    model = build(gram_3)
    # make_plot_prior(model)
    # make_plot_post(model)
    make_leanring_plot(model)
    # for c in model_3.nodes["TH"].right_children:
    #     print(c.gram)
    # print(model_3.nodes["HE"].left_children.gram)
    # model_3.inference()
    # model_3.nodes["TH"].print_results()
    # model_3.nodes["HE"].print_results()
    # model_3.nodes["T"].print_results()
    # model_3.nodes["WT"].print_results()
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