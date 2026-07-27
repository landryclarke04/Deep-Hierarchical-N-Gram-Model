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
    grams = []
    # starting with 5
    all_grams = model.nodes.keys()
    # getting no tri-grams
    no_tri_grams = [g for g in all_grams if len(g) != 3]
    unique_numbers = random.sample(range(0, len(no_tri_grams)), 15)
    for i in range(len(unique_numbers)):
        grams.append(no_tri_grams[unique_numbers[i]])
    
    grams_mod = []
    num_children = []
    x = grams_mod
    
    labels = grams_mod
    for g in grams:
        
        node = model.nodes[g]
        num_children.append(node.get_number_of_children())
        i = random.randint(0, node.p-1)
        # i = 0
        g += "[" + str(i) + "]"
        grams_mod.append(g)
        mean = node.mean_copies[0].flatten()[i]
        prior_means.append(mean)
        true_mean = node.get_true_mean().flatten()[i]
        #prior_std_scalar = [np.sqrt(Sigma[0, 0]) for Sigma in covariances]
        # prior_std = np.sqrt((node.prior_var[i][i]))
        post_std = abs(np.quantile(np.array(node.mean_copies), [0.975])[0])
        # post_std = np.std(np.array(node.mean_copies), axis=0)[i][0]
        prior_std = post_std
        #prior_std_scalar = [np.sqrt(Sigma[0, 0]) for Sigma in covariances]
        # prior_std = np.sqrt((node.prior_var[i][i]))
        # prior_std = abs((0.5+mean) - (mean-0.5))/4
        # prior_std = 0.1 * abs(node.mean_copies[0].flatten()[i])
        # print(np.sqrt(np.diag(node.prior_var)))
        prior_stds.append(prior_std)
        true_means.append(node.get_true_mean().flatten()[i])
        post_means.append(node.get_est_mean().flatten()[i])
        #prior_std_scalar = [np.sqrt(Sigma[0, 0]) for Sigma in covariances]
        # post_std = np.sqrt((node.post_var[i][i]))
        
        # prior_std = 0.1 * abs(node.mean_copies[0].flatten()[i])
        # print(np.sqrt(np.diag(node.prior_var)))
        post_stds.append(post_std)

    sort_idx = np.argsort(true_means)[::-1] 
    true_means      = np.array(true_means)[sort_idx]
    prior_means     = np.array(prior_means)[sort_idx]
    prior_stds      = np.array(prior_stds)[sort_idx]
    post_means = np.array(post_means)[sort_idx]
    post_stds  = np.array(post_stds)[sort_idx]
    num_children = np.array(num_children)[sort_idx]
    grams = np.array(grams)[sort_idx]

    labels = np.array(labels)[sort_idx]
    x = np.arange(len(labels))
    offset = 0.15

    ax.errorbar(
        x - offset,
        prior_means,
        yerr=prior_stds,
        fmt='o',
        capsize=4,
        label='Initial Mean ±2σ'
    )

    ax.errorbar(
        x + offset,
        post_means,
        yerr=post_stds,
        fmt='s',
        capsize=4,
        label='Posterior ±2σ'
    )

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
    ax.legend(
        # loc='center left',
        # bbox_to_anchor=(1.02, 0.5)
    )
    ax.set_title(
        "Learning Phase:\n"
        "[i] correspond to component of mean chosen starting at 0",
        fontsize=10
    )

    table_data = [
        # ['Blue',   'm = 10'],
        # ['Orange', 'm = 50'],
        # ['Green',  'm = 100'],
        # ['Red',    'm = 500'],
        # ['Purple', 'm = 1000']
    ]

    for i in range(len(grams)):
        row = [grams[i], str(num_children[i])]
        table_data.append(row)

    table = plt.table(
        cellText=table_data,
        colLabels=['Gram', 'Number of Children'],
        loc='center left',
        bbox=[1.05, 0.25, 0.3, 0.5],
        cellLoc='center'
    )
    table.set_fontsize(12)

    table.scale(1.2, 1.7)

    plt.tight_layout()
    plt.show()

def make_M_plot(model):
    fig, ax = plt.subplots(figsize=(10, 6))

    prior_means = []
    prior_stds = []
    true_means = []
    post_means = []
    post_stds = []
    m_values = []
    # starting with 5
    # grams = ["HER", "THL", "HEM", "THW", "THE"]
    all_grams = model.nodes.keys()
    # getting only tri-grams
    tri_grams = [g for g in all_grams if len(g) == 3]
    grams = []
    for m in Node.M:
        grams += M_helper(tri_grams, m, model)
    grams_mod = []
    x = grams_mod

    
    labels = grams_mod
    for g in grams:
        
        node = model.nodes[g]
        i = random.randint(0, node.p-1)
        # i = 0
        g += "[" + str(i) + "]"
        grams_mod.append(g)
        # mean = node.get_prior_mean().flatten()[i]
        mean = node.mean_copies[0].flatten()[i]
        prior_means.append(mean)
        true_mean = node.get_true_mean().flatten()[i]
        #prior_std_scalar = [np.sqrt(Sigma[0, 0]) for Sigma in covariances]
        # prior_std = np.sqrt((node.prior_var[i][i]))
        s = abs(mean-true_mean)
        # if node.get_m() != 0:
        #     s = s/math.sqrt(node.get_m())
        post_std = abs(np.quantile(np.array(node.mean_copies), [0.975])[0])
        # post_std = np.std(np.array(node.mean_copies), axis=0)[i][0]
        prior_std = post_std
        #prior_std_scalar = [np.sqrt(Sigma[0, 0]) for Sigma in covariances]
        # prior_std = np.sqrt((node.prior_var[i][i]))
        # prior_std = abs((0.5+mean) - (mean-0.5))/4
        # prior_std = 0.1 * abs(node.mean_copies[0].flatten()[i])
        # print(np.sqrt(np.diag(node.prior_var)))
        prior_stds.append(prior_std)
        true_means.append(node.get_true_mean().flatten()[i])
        post_means.append(node.get_est_mean().flatten()[i])
        #prior_std_scalar = [np.sqrt(Sigma[0, 0]) for Sigma in covariances]
        # post_std = np.sqrt((node.post_var[i][i]))
        
        # prior_std = 0.1 * abs(node.mean_copies[0].flatten()[i])
        # print(np.sqrt(np.diag(node.prior_var)))
        post_stds.append(post_std)
        m_values.append(node.get_m())

    m_colors = {
        0: (2/255, 8/255, 106/255),
        2: (144/255, 41/255, 43/255),
        10: (208/255, 44/255, 129/255),
        100: (0/255, 159/255, 136/255), 
        500: (91/255, 47/255, 110/255)
    }

    # sort_idx = np.argsort(true_means)[::-1] 
    sort_idx = np.argsort(m_values)
    m_values        = np.array(m_values)[sort_idx]
    true_means      = np.array(true_means)[sort_idx]
    prior_means     = np.array(prior_means)[sort_idx]
    prior_stds      = np.array(prior_stds)[sort_idx]
    post_means = np.array(post_means)[sort_idx]
    post_stds  = np.array(post_stds)[sort_idx]

    labels = np.array(labels)[sort_idx]
    x = np.arange(len(labels))
    offset = 0.2

    ax.errorbar(
        x - offset,
        prior_means,
        yerr=prior_stds,
        fmt='o',
        capsize=4,
        label='Initial Mean ±2σ'
    )

    ax.errorbar(
        x + offset,
        post_means,
        yerr=post_stds,
        fmt='s',
        capsize=4,
        label='Posterior Mean ±2σ'
    )

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
    for tick, m in zip(ax.get_xticklabels(), m_values):
        tick.set_color(m_colors[m])
    legend_handles = [
        Line2D([0], [0], color=color, lw=4, label=f'm = {m}')
        for m, color in m_colors.items()
    ]
    # ax.legend(
    #     loc='center left',
    #     bbox_to_anchor=(1.02, 0.5)
    # )

    ax.legend(handles=ax.get_legend_handles_labels()[0] + legend_handles,
                loc='center left',
                bbox_to_anchor=(1.02, 0.5)
    )
    # ax.legend()
    ax.set_xlabel("Tri-gram where [i] denotes the index of a randomly chosen component of the mean vector, starting at 0")
    ax.set_ylabel("Mean Value")
    ax.set_title(
        "Learning Phase for Tri-grams Where Sample Size, m, Varies",
        fontsize=10
    )
   

    plt.tight_layout()
    plt.show()
    return

def M_helper(tri_grams, m, model):
    m_gram = []
    i = 0
    while len(m_gram) < 2:
        i = random.randint(0, len(tri_grams)-1)
        if model.nodes[tri_grams[i]].m == m:
            if len(m_gram) > 0:
                if tri_grams[i] != m_gram[0]:
                    m_gram.append(tri_grams[i])
            else:
                m_gram.append(tri_grams[i])
            
            # tri_grams.pop(i)
    return m_gram

def make_predictive_plot(model):
    fig, ax = plt.subplots(figsize=(10, 6))
    grams = ["T", "H", "E", "TH", "HE", "THE"]
    stds = []
    mod_grams = []
    true_means = []
    
    col_y = []
    for g in grams:
        
        for i in range(0, len(g)*2-1, 2):
            new_y = []
            mod_grams.append(g + "[" + str(i) + "]")
            
            
            node = model.nodes[g]
            true_means.append(node.get_true_mean().flatten()[i])
            for mu in node.mean_copies:
                y = node.sample_MVN(mu, node.true_var)
                new_y.append(y.flatten().copy()[i])
            std = abs(np.quantile(np.array(new_y), [0.975])[0])
            # print(std)
            # stds.append(2*np.std(np.array(new_y), axis=0)[i])
            stds.append(std)
            mean_pred = np.array(new_y).mean(axis=0)
            col_y.append(mean_pred)

    labels = mod_grams
    x = np.arange(len(labels))

    ax.errorbar(
        x,
        col_y,
        yerr=stds,
        color = (91/255, 47/255, 110/255),
        fmt='o',
        capsize=4,
        label='Preditive Posterior Mean ±2σ'
    )

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
    # ax.legend(
    #     loc='center left',
    #     bbox_to_anchor=(1.02, 0.5)
    # )

    ax.legend()
    ax.set_title(
        "Predictive Posterior Mean Compared to True Mean Focused on \"THE\":\n"
        "Taken after adding all possible uppercase tri-grams (total of 104) containing \"TH\" and \"HE\"",
        fontsize=10
    )
    # fig.text(
    #     0.85, 0.5,
    #     "Notes:\n"
    #     "• Colored x-axis labels indicate sample size m.\n"
    #     "• Values correspond to component i=2 of each mean vector.\n"
    #     "• Error bars represent ±2 standard deviations.",
    #     fontsize=9,
    #     va='center'
    # )
    ax.set_xlabel("Possible n-grams made from tri-gram \"THE\"")
    ax.set_ylabel("Mean Value")

    plt.tight_layout()
    plt.show()
    return

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
    make_one_learning_plot(model)
    return model

def main():
    gram_3 = "THE"
    model = build(gram_3)
    # for g in model.nodes:
    #     model.nodes[g].print_results()

    
    # make_one_learning_plot(model)
    make_M_plot(model)
    # make_predictive_plot(model)


    # build_full()
    
    return 0
main()