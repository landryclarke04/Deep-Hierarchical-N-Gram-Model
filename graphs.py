import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import invwishart

import random

import string

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from mean_general import Node
from model import Model

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
       
        g += "[" + str(i) + "]"
        grams_mod.append(g)
        mean = node.get_marginal_mean().flatten()[i]
        prior_means.append(mean)
       
        post_std = abs(np.quantile(np.array(node.mean_copies), [0.975])[0])
        
        prior_std = abs(np.quantile(np.array(node.marginal_priors), [0.975])[0])
       
        prior_stds.append(prior_std)
        true_means.append(node.get_true_mean().flatten()[i])
        post_means.append(node.get_est_mean().flatten()[i])
       
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
       
    )
    ax.set_title(
        "Learning Phase:\n"
        "[i] correspond to component of mean chosen starting at 0",
        fontsize=10
    )

    table_data = []

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
        mean = node.get_marginal_mean().flatten()[i]
        prior_means.append(mean)
        post_std = abs(np.quantile(np.array(node.mean_copies), [0.975])[0])
        # post_std = np.std(np.array(node.mean_copies), axis=0)[i][0]
        prior_std = abs(np.quantile(np.array(node.marginal_priors), [0.975])[0])
        prior_stds.append(prior_std)
        true_means.append(node.get_true_mean().flatten()[i])
        post_means.append(node.get_est_mean().flatten()[i])

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

    tri_grams = [g for g in model.nodes.keys() if len(g) == 3]
    grams = []
    for m in Node.M:
        grams += M_helper(tri_grams, m, model)
    stds = []
    mod_grams = []
    true_means = []
    m_values = []
    prior_stds = []
    prior_means = []
    new_samples = []
    
    col_y = []
    for g in grams:
        node = model.nodes[g]
        i = random.randint(0, node.p-1)

        # new not trained on data
        new_node_samples = []
        for j in range(10):
            new_node_samples.append(node.sample_MVN(node.true_mean, node.true_var).flatten()[i])
        new_samples.append(new_node_samples)
        new_y = []
        m_values.append(node.get_m())
        mod_grams.append(g + "[" + str(i) + "]")

        burn = int(len(node.mean_copies)*.1)
        # predictive priors
        prior_mean = np.array(node.marginal_data[burn:]).mean(axis=0).flatten()[i]
        prior_means.append(prior_mean)
        prior_std = abs(np.quantile(np.array(node.marginal_data[burn:]), [0.975])[0])
        prior_stds.append(prior_std)
        
        true_means.append(node.get_true_mean().flatten()[i])
        
        for mu in node.mean_copies[burn:]:
            y = node.sample_MVN(mu, node.true_var)
            new_y.append(y.flatten().copy()[i])
        std = abs(np.quantile(np.array(new_y), [0.975])[0])
        # print(std)
        # stds.append(2*np.std(np.array(new_y), axis=0)[i])
        stds.append(std)
        mean_pred = np.array(new_y).mean(axis=0)
        col_y.append(mean_pred)

    m_colors = {
        0: (2/255, 8/255, 106/255),
        2: (144/255, 41/255, 43/255),
        10: (208/255, 44/255, 129/255),
        100: (0/255, 159/255, 136/255), 
        500: (91/255, 47/255, 110/255)
    }
    labels = mod_grams

    # sort_idx = np.argsort(true_means)[::-1] 
    sort_idx = np.argsort(m_values)
    m_values        = np.array(m_values)[sort_idx]
    true_means      = np.array(true_means)[sort_idx]
    col_y = np.array(col_y)[sort_idx]
    stds  = np.array(stds)[sort_idx]
    prior_stds = np.array(prior_stds)[sort_idx]
    prior_means = np.array(prior_means)[sort_idx]
    new_samples = np.array(new_samples)[sort_idx]

    labels = np.array(labels)[sort_idx]
    x = np.arange(len(labels))
    offset = 0.15

    ax.errorbar(
        x + offset,
        prior_means,
        yerr=prior_stds,
        fmt='o',
        capsize=4,
        label='Initial Mean ±2σ'
    )

    ax.errorbar(
        x-offset,
        col_y,
        yerr=stds,
        # color = (91/255, 47/255, 110/255),
        fmt='o',
        capsize=4,
        label='Preditive Posterior Mean ±2σ'
    )

    # Truth
    for i, samples in enumerate(new_samples):
        jitter = np.random.uniform(-0.08, 0.08, len(samples))

        ax.scatter(
            np.full(len(samples), x[i]),
            samples,
            marker='*',
            color = "red",
            s=40,
            alpha=0.7,
            label='New Samples' if i == 0 else ""
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    for tick, m in zip(ax.get_xticklabels(), m_values):
        tick.set_color(m_colors[m])
    legend_handles = [
        Line2D([0], [0], color=color, lw=4, label=f'm = {m}')
        for m, color in m_colors.items()
    ]

    ax.legend(handles=ax.get_legend_handles_labels()[0] + legend_handles,
                loc='center left',
                bbox_to_anchor=(1.02, 0.5)
    )

    ax.set_title(
        "Predictive Posterior Mean Compared to New Sample Data",
        fontsize=10
    )
  
    ax.set_xlabel("Tri-gram where [i] denotes the index of a randomly chosen component of the mean vector, starting at 0")
    ax.set_ylabel("Mean Value")
   

    plt.tight_layout()
    plt.show()
    return

def make_M_and_predictive(model):
    fig, ax = plt.subplots(figsize=(10, 6))

    prior_means = []
    prior_stds = []
    true_means = []
    post_means = []
    post_stds = []
    m_values = []
    stds = []
    new_samples = []
    marg_means = []
    
    col_y = []
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
        burn = int(len(node.mean_copies)*.1)
        # i = 0
        g += "[" + str(i) + "]"
        grams_mod.append(g)
        # mean = node.get_prior_mean().flatten()[i]
        mean = np.array(node.marginal_priors[burn:]).mean(axis=0).flatten()[i]
        marg_mean = node.get_marginal_mean().flatten()[i]
        marg_means.append(marg_mean)
        prior_means.append(mean)
        post_std = abs(np.quantile(np.array(node.mean_copies[burn:]), [0.975])[0])
        # post_std = np.std(np.array(node.mean_copies), axis=0)[i][0]
        prior_std = abs(np.quantile(np.array(node.marginal_priors[burn:]), [0.975])[0])
        prior_stds.append(prior_std)
        true_means.append(node.get_true_mean().flatten()[i])
        post_means.append(node.get_est_mean().flatten()[i])

        post_stds.append(post_std)
        m_values.append(node.get_m())
        # predictive part
        new_node_samples = []
        for j in range(10):
            new_node_samples.append(node.sample_MVN(node.true_mean, node.true_var).flatten()[i])
        new_samples.append(new_node_samples)
        new_y = []
        
        for mu in node.mean_copies[burn:]:
            y = node.sample_MVN(mu, node.true_var)
            new_y.append(y.flatten().copy()[i])
        std = abs(np.quantile(np.array(new_y), [0.975])[0])
        # print(std)
        # stds.append(2*np.std(np.array(new_y), axis=0)[i])
        stds.append(std)
        mean_pred = np.array(new_y).mean(axis=0)
        col_y.append(mean_pred)

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
    marg_means = np.array(marg_means)[sort_idx]

    col_y = np.array(col_y)[sort_idx]
    stds  = np.array(stds)[sort_idx]

    labels = np.array(labels)[sort_idx]
    x = np.arange(len(labels))
    offset = 0.3
    sub_offest = 0.1

    ax.errorbar(
        x - offset,
        prior_means,
        yerr=prior_stds,
        fmt='o',
        capsize=4,
        label='Prior Mean ±2σ'
    )

    ax.errorbar(
        x + offset,
        post_means,
        yerr=post_stds,
        fmt='s',
        capsize=4,
        label='Posterior Mean ±2σ'
    )

    ax.scatter(
        x-sub_offest,
        marg_means,
        marker = 'd',
        color = (255/255,215/255,0),
        s=100,
        label = 'Initial Mean',
        zorder=1
    )

    # Truth
    ax.scatter(
        x + sub_offest,
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

    fig, ax = plt.subplots(figsize=(10, 6))


    offset = 0.17

    ax.errorbar(
        x - offset,
        prior_means,
        yerr=prior_stds,
        fmt='o',
        capsize=4,
        label='Prior  Mean ±2σ'
    )

    ax.errorbar(
        x+offset,
        col_y,
        yerr=stds,
        # color = (91/255, 47/255, 110/255),
        fmt='o',
        capsize=4,
        label='Preditive Posterior Mean ±2σ'
    )

    # Truth
    for i, samples in enumerate(new_samples):
        jitter = np.random.uniform(-0.08, 0.08, len(samples))

        ax.scatter(
            np.full(len(samples), x[i]),
            samples,
            marker='*',
            color = "red",
            s=40,
            alpha=0.7,
            label='New Samples from True Paramters' if i == 0 else ""
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    for tick, m in zip(ax.get_xticklabels(), m_values):
        tick.set_color(m_colors[m])
    legend_handles = [
        Line2D([0], [0], color=color, lw=4, label=f'm = {m}')
        for m, color in m_colors.items()
    ]

    ax.legend(handles=ax.get_legend_handles_labels()[0] + legend_handles,
                loc='center left',
                bbox_to_anchor=(1.02, 0.5)
    )

    ax.set_title(
        "Predictive Posterior Mean Compared to New Sample Data",
        fontsize=10
    )
  
    ax.set_xlabel("Tri-gram where [i] denotes the index of a randomly chosen component of the mean vector, starting at 0")
    ax.set_ylabel("Mean Value")
   

    plt.tight_layout()
    plt.show()