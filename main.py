import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import invwishart

import random

import string

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
    #     model.add_gram(l + gram[1:])
    # # print(model.nodes["ITH"].y)
    model.inference()
    model.nodes["TH"].print_results()
    model.nodes["HE"].print_results()
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
    grams = ["T", "TH", "HE", "H", "THE", "E"]
    grams_mod = []
    x = grams_mod
    labels = grams_mod
    for g in grams:
        
        node = model.nodes[g]
        # i = random.randint(0, node.p-1)
        i = 0
        g += "[" + str(i) + "]"
        grams_mod.append(g)
        mean = node.get_prior_mean().flatten()[i]
        prior_means.append(mean)
        true_mean = node.get_true_mean().flatten()[i]
        #prior_std_scalar = [np.sqrt(Sigma[0, 0]) for Sigma in covariances]
        # prior_std = np.sqrt((node.prior_var[i][i]))
        prior_std = abs(mean-true_mean)/2
        # prior_std = 0.1 * abs(node.mean_copies[0].flatten()[i])
        # print(np.sqrt(np.diag(node.prior_var)))
        prior_stds.append(2*prior_std)
        true_means.append(true_mean)
        post_means.append(node.get_est_mean().flatten()[i])
        #prior_std_scalar = [np.sqrt(Sigma[0, 0]) for Sigma in covariances]
        # post_std = np.sqrt((node.post_var[i][i]))
        post_std = np.std(np.array(node.mean_copies), axis=0)[i][0]
        # prior_std = 0.1 * abs(node.mean_copies[0].flatten()[i])
        # print(np.sqrt(np.diag(node.prior_var)))
        post_stds.append(2*post_std)

    sort_idx = np.argsort(true_means)[::-1] 
    true_means      = np.array(true_means)[sort_idx]
    prior_means     = np.array(prior_means)[sort_idx]
    prior_stds      = np.array(prior_stds)[sort_idx]
    post_means = np.array(post_means)[sort_idx]
    post_stds  = np.array(post_stds)[sort_idx]

    labels = np.array(labels)[sort_idx]

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
    handles, labels = ax1.get_legend_handles_labels()

    # extra = Line2D(
    #     [0], [0],
    #     color='none',
    #     label='Dimension=[i]'
    # )


    ax2.set_title('Posterior')
    # ax2.set_xticks(x)
    # ax2.set_xticklabels(labels, rotation=45)

    plt.tight_layout()
    plt.legend()
    # ax1.legend()
    fig.suptitle(
    f"Prior and Posterior Comparison (mean component [index])"
    )
    ax1.set_visible(False)
    plt.show()

def make_one_learning_plot(model):
    fig, ax = plt.subplots(figsize=(10, 6))

    prior_means = []
    prior_stds = []
    true_means = []
    post_means = []
    post_stds = []
    # starting with 5
    grams = ["T", "TH", "HE", "H", "THE", "E"]
    grams_mod = []
    x = grams_mod
    
    labels = grams_mod
    for g in grams:
        
        node = model.nodes[g]
        # i = random.randint(0, node.p-1)
        i = 0
        g += "[" + str(i) + "]"
        grams_mod.append(g)
        mean = node.get_prior_mean().flatten()[i]
        prior_means.append(mean)
        true_mean = node.get_true_mean().flatten()[i]
        #prior_std_scalar = [np.sqrt(Sigma[0, 0]) for Sigma in covariances]
        # prior_std = np.sqrt((node.prior_var[i][i]))
        prior_std = abs(mean-true_mean)/2
        #prior_std_scalar = [np.sqrt(Sigma[0, 0]) for Sigma in covariances]
        # prior_std = np.sqrt((node.prior_var[i][i]))
        # prior_std = abs((0.5+mean) - (mean-0.5))/4
        # prior_std = 0.1 * abs(node.mean_copies[0].flatten()[i])
        # print(np.sqrt(np.diag(node.prior_var)))
        prior_stds.append(2*prior_std)
        true_means.append(node.get_true_mean().flatten()[i])
        post_means.append(node.get_est_mean().flatten()[i])
        #prior_std_scalar = [np.sqrt(Sigma[0, 0]) for Sigma in covariances]
        # post_std = np.sqrt((node.post_var[i][i]))
        post_std = np.std(np.array(node.mean_copies), axis=0)[i][0]
        # prior_std = 0.1 * abs(node.mean_copies[0].flatten()[i])
        # print(np.sqrt(np.diag(node.prior_var)))
        post_stds.append(2*post_std)

    sort_idx = np.argsort(true_means)[::-1] 
    true_means      = np.array(true_means)[sort_idx]
    prior_means     = np.array(prior_means)[sort_idx]
    prior_stds      = np.array(prior_stds)[sort_idx]
    post_means = np.array(post_means)[sort_idx]
    post_stds  = np.array(post_stds)[sort_idx]

    labels = np.array(labels)[sort_idx]
    x = np.arange(len(labels))
    offset = 0.15

    ax.errorbar(
        x - offset,
        prior_means,
        yerr=2*prior_stds,
        fmt='o',
        capsize=4,
        label='Prior'
    )

    ax.errorbar(
        x + offset,
        post_means,
        yerr=2*post_stds,
        fmt='s',
        capsize=4,
        label='Posterior'
    )

    # # Prior
    # ax.errorbar(
    #     x,
    #     prior_means,
    #     yerr=2*prior_stds,
    #     fmt='o',
    #     capsize=4,
    #     label='Prior Mean ± 2σ'
    # )

    # # Posterior
    # ax.errorbar(
    #     x,
    #     post_means,
    #     yerr=2*post_stds,
    #     fmt='s',
    #     capsize=4,
    #     label='Posterior Mean ± 2σ'
    # )

    # Truth
    ax.scatter(
        x,
        true_means,
        marker='*',
        color = "red",
        s=150,
        label='True Mean',
        zorder=5
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.show()

def main():
    # gram_3 = "THE"
    # gram_3 = "EXODUS"
    gram_3 = "THE"

    # model_3 = test_build(gram_3, 500)
    model = build(gram_3)
    # make_plot_prior(model)
    # make_plot_post(model)
    # make_leanring_plot(model)
    make_one_learning_plot(model)
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