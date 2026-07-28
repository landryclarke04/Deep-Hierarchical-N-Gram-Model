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
    gibbs_inits = []
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
        mean = node.get_marginal_prior_mean().flatten()[i]
        prior_means.append(mean)
        gibbs_inits.append(node.get_gibbs_init_mean().flatten()[i])

        post_std = abs(np.quantile(np.array(node.mean_copies), [0.975])[0])

        # ±2σ spread of the marginal-prior samples for this component
        prior_std = 2 * np.std(np.array(node.marginal_priors)[:, i, 0])

        prior_stds.append(prior_std)
        true_means.append(node.get_true_mean().flatten()[i])
        post_means.append(node.get_est_mean().flatten()[i])
       
        post_stds.append(post_std)

    sort_idx = np.argsort(true_means)[::-1]
    true_means      = np.array(true_means)[sort_idx]
    prior_means     = np.array(prior_means)[sort_idx]
    prior_stds      = np.array(prior_stds)[sort_idx]
    gibbs_inits     = np.array(gibbs_inits)[sort_idx]
    post_means = np.array(post_means)[sort_idx]
    post_stds  = np.array(post_stds)[sort_idx]
    num_children = np.array(num_children)[sort_idx]
    grams = np.array(grams)[sort_idx]

    labels = np.array(labels)[sort_idx]
    x = np.arange(len(labels))
    offset = 0.18

    # Blue: mean of the marginal-prior samples (the "prior marginal mean")
    ax.errorbar(
        x - 2*offset,
        prior_means,
        yerr=prior_stds,
        fmt='o',
        capsize=4,
        label='Prior Marginal Mean ±2σ'
    )

    # Black cross: the single marginal-prior draw that seeds Gibbs (est_mean at t=0)
    ax.scatter(
        x - offset,
        gibbs_inits,
        marker='X',
        color='black',
        s=70,
        label='Gibbs Initial Value',
        zorder=6
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
    gibbs_inits = []
    true_means = []
    post_means = []
    post_stds = []
    m_values = []
    stds = []
    new_samples = []

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
        mean = node.get_marginal_prior_mean().flatten()[i]
        prior_means.append(mean)
        gibbs_inits.append(node.get_gibbs_init_mean().flatten()[i])
        post_std = abs(np.quantile(np.array(node.mean_copies[burn:]), [0.975])[0])
        # post_std = np.std(np.array(node.mean_copies), axis=0)[i][0]
        # ±2σ spread of the marginal-prior samples for this component
        prior_std = 2 * np.std(np.array(node.marginal_priors)[burn:, i, 0])
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
    gibbs_inits     = np.array(gibbs_inits)[sort_idx]
    post_means = np.array(post_means)[sort_idx]
    post_stds  = np.array(post_stds)[sort_idx]

    col_y = np.array(col_y)[sort_idx]
    stds  = np.array(stds)[sort_idx]

    labels = np.array(labels)[sort_idx]
    x = np.arange(len(labels))
    offset = 0.18

    # Blue: mean of the marginal-prior samples (the "prior marginal mean")
    ax.errorbar(
        x - 2*offset,
        prior_means,
        yerr=prior_stds,
        fmt='o',
        capsize=4,
        label='Prior Marginal Mean ±2σ'
    )

    # Black cross: the single marginal-prior draw that seeds Gibbs (est_mean at t=0)
    ax.scatter(
        x - offset,
        gibbs_inits,
        marker='X',
        color='black',
        s=70,
        label='Gibbs Initial Value',
        zorder=6
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

    fig, ax = plt.subplots(figsize=(10, 6))


    offset = 0.15

    ax.errorbar(
        x + offset,
        prior_means,
        yerr=prior_stds,
        fmt='o',
        capsize=4,
        label='Prior Marginal Mean ±2σ'
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

# ---------------------------------------------------------------------------
# Error-based redesign: highlight the effectiveness of learning by plotting
# error relative to truth (truth == 0), not absolute mean levels.
# ---------------------------------------------------------------------------

def _collect_errors(model):
    """Prior- and posterior-error for every component of every fitted node.

    Returns (prior_err, post_err) as flat arrays of (estimate - true), where
    the prior estimate is the marginal-prior mean and the posterior estimate
    is the Gibbs posterior mean (est_mean after burn-in averaging).
    """
    prior_err = []
    post_err = []
    for gram, node in model.nodes.items():
        tm = node.get_true_mean()
        pm = node.get_marginal_prior_mean()
        em = node.get_est_mean()
        if tm is None or pm is None or em is None:
            continue
        tm = tm.flatten(); pm = pm.flatten(); em = em.flatten()
        prior_err.extend((pm - tm).tolist())
        post_err.extend((em - tm).tolist())
    return np.array(prior_err), np.array(post_err)


def make_learning_error_plot(model, n_show=18):
    """Two-panel learning figure.

    Left  : per-gram dumbbell of error (prior -> posterior), truth at 0.
    Right : ECDF of |error| over ALL components (prior vs posterior).
    """
    # ----- aggregate over every component -----
    all_prior_err, all_post_err = _collect_errors(model)

    # ----- sample a readable subset for the per-gram panel -----
    fitted = [g for g in model.nodes if model.nodes[g].get_est_mean() is not None]
    chosen = random.sample(fitted, min(n_show, len(fitted)))
    rows = []
    for g in chosen:
        node = model.nodes[g]
        i = random.randint(0, node.p - 1)
        tm = node.get_true_mean().flatten()[i]
        pe = node.get_marginal_prior_mean().flatten()[i] - tm
        po = node.get_est_mean().flatten()[i] - tm
        # estimator standard deviations (invariant to the -true shift).
        # prior: spread of the marginal-prior samples; posterior: spread of
        # the (post burn-in) Gibbs samples of the mean.
        mp = np.array(node.marginal_priors)[:, i, 0]
        mc = np.array(node.mean_copies)
        burn = int(len(mc) * 0.1)
        mc = mc[burn:, i, 0]
        rows.append((f"{g}[{i}]", pe, po, mp.std(), mc.std()))
    # largest prior error at the top
    rows.sort(key=lambda r: abs(r[1]))
    labels = [r[0] for r in rows]
    prior_e = np.array([r[1] for r in rows])
    post_e = np.array([r[2] for r in rows])
    prior_sd = np.array([r[3] for r in rows])
    post_sd = np.array([r[4] for r in rows])
    y = np.arange(len(rows))

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1.6, 1]}
    )

    # ===== Panel A: per-gram error, prior -> posterior =====
    # error is on the x-axis, so the +/-2sigma credible intervals are xerr.
    # A bar straddling the red zero line means the estimator's interval
    # covers the truth.
    axL.axvline(0, color="red", lw=1.5, zorder=1, label="Truth (error = 0)")
    for yi, pe, po in zip(y, prior_e, post_e):
        axL.annotate(
            "", xy=(po, yi), xytext=(pe, yi),
            arrowprops=dict(arrowstyle="->", color="0.6", lw=1.8), zorder=2,
        )
    axL.errorbar(prior_e, y, xerr=2 * prior_sd, fmt="o", color="tab:blue",
                 capsize=3, markersize=6, zorder=3,
                 label="Prior marginal mean  ±2σ")
    axL.errorbar(post_e, y, xerr=2 * post_sd, fmt="s", color="tab:green",
                 capsize=3, markersize=6, zorder=3,
                 label="Posterior mean  ±2σ")
    axL.set_yticks(y)
    axL.set_yticklabels(labels, fontsize=8)
    axL.set_xlabel("Estimate − True   (0 = perfect)")
    axL.set_title("Per-gram error ±2σ  (bar crossing 0 ⇒ interval covers truth)",
                  fontsize=10)
    axL.legend(fontsize=8, loc="lower right")

    # ===== Panel B: aggregate |error| ECDF over all components =====
    def ecdf(a):
        s = np.sort(a)
        return s, np.arange(1, len(s) + 1) / len(s)

    xp, yp = ecdf(np.abs(all_prior_err))
    xq, yq = ecdf(np.abs(all_post_err))
    axR.plot(xp, yp, color="tab:blue", lw=2,
             label=f"Prior  (median {np.median(np.abs(all_prior_err)):.2f})")
    axR.plot(xq, yq, color="tab:green", lw=2,
             label=f"Posterior  (median {np.median(np.abs(all_post_err)):.2f})")
    axR.set_xlabel("|Estimate - True|")
    axR.set_ylabel("Fraction of components <= x")
    axR.set_title(f"Absolute error over all {len(all_prior_err)} components")
    axR.legend(fontsize=9, loc="lower right")

    beats = float(np.mean(np.abs(all_post_err) < np.abs(all_prior_err))) * 100
    fig.suptitle(
        "Effectiveness of learning: posterior beats prior for "
        f"{beats:.0f}% of components",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


def make_error_vs_m_plot(model):
    """Learning vs. sample size, aggregated over all tri-grams.

    For every tri-gram we compute the RMSE (over that gram's mean
    components) of three estimators:
      - prior marginal mean : never sees data -> flat baseline
      - Gibbs initial value : a single marginal-prior draw -> flat, but
                              noisier than the prior *mean*
      - Gibbs posterior mean: conditions on the data -> falls sharply in m

    Each curve is the median RMSE across tri-grams with an inter-quartile
    band (robust to the right-skew of RMSE; never dips below 0). The y-axis
    is log-scaled because the posterior spans roughly two decades.
    """
    from collections import defaultdict

    def rmse(est, true):
        return float(np.sqrt(np.mean((est.flatten() - true.flatten()) ** 2)))

    # (attribute getter, color, line style, legend label)
    curves = {
        "prior":     ("get_marginal_prior_mean", "tab:blue",   "o--",
                      "Prior marginal mean (baseline)"),
        "init":      ("get_gibbs_init_mean",     "tab:orange", "^:",
                      "Gibbs initial value (single prior draw)"),
        "posterior": ("get_est_mean",            "tab:green",  "s-",
                      "Gibbs posterior mean"),
    }

    by_m = {k: defaultdict(list) for k in curves}
    for g in model.nodes:
        if len(g) != 3:
            continue
        node = model.nodes[g]
        tm = node.get_true_mean()
        if tm is None:
            continue
        m = node.get_m()
        for k, (getter, *_rest) in curves.items():
            est = getattr(node, getter)()
            if est is not None:
                by_m[k][m].append(rmse(est, tm))

    ms = sorted(by_m["posterior"])
    xpos = np.arange(len(ms))

    fig, ax = plt.subplots(figsize=(9, 6))
    for k, (getter, color, fmt, label) in curves.items():
        med = np.array([np.median(by_m[k][m]) for m in ms])
        q25 = np.array([np.percentile(by_m[k][m], 25) for m in ms])
        q75 = np.array([np.percentile(by_m[k][m], 75) for m in ms])
        ax.fill_between(xpos, q25, q75, color=color, alpha=0.15)
        ax.plot(xpos, med, fmt, color=color, lw=2, label=label)

    ax.set_yscale("log")
    ax.set_xticks(xpos)
    ax.set_xticklabels([str(m) for m in ms])
    ax.set_xlabel("m  (observations per tri-gram)")
    ax.set_ylabel("RMSE per tri-gram  (median, IQR band; log scale)")
    ax.set_title("Learning vs. sample size: posterior RMSE falls sharply with m")
    ax.legend()
    plt.tight_layout()
    plt.show()


def make_trigram_learning_plot(model):
    """Learning phase for tri-grams: representative error vs sample size m.

    One tick per value of m. For each m we pick a *representative*
    tri-gram+component -- the one whose posterior-mean approximation error
    is the median over all tri-gram+component pairs with that m (so it is
    neither the best nor worst case). At each tick we plot the approximation
    error (estimate - true) for two estimators, each with its +/-2 sigma
    band:
        (a) prior marginal mean   (blue)
        (b) Gibbs posterior mean  (green)

    The bars are the standard error of the *approximation error* (identical
    to the estimator's standard error, since -true is a constant shift). A
    good estimator's interval should cover 0, and its width should shrink as
    m grows.
    """
    from collections import defaultdict

    # ----- gather every (|posterior error|, gram, component) per m -----
    cand = defaultdict(list)
    for g in model.nodes:
        if len(g) != 3:
            continue
        node = model.nodes[g]
        tm = node.get_true_mean()
        em = node.get_est_mean()
        if tm is None or em is None:
            continue
        tm = tm.flatten(); em = em.flatten()
        m = node.get_m()
        for i in range(len(tm)):
            cand[m].append((abs(em[i] - tm[i]), g, i))

    ms = sorted(cand)

    # ----- pick the median-error representative for each m -----
    labels = []
    prior_e = []; prior_sd = []
    post_e = []; post_sd = []
    for m in ms:
        lst = sorted(cand[m], key=lambda r: r[0])
        _, g, i = lst[len(lst) // 2]          # median by |posterior error|
        node = model.nodes[g]
        tm = node.get_true_mean().flatten()[i]
        prior_e.append(node.get_marginal_prior_mean().flatten()[i] - tm)
        post_e.append(node.get_est_mean().flatten()[i] - tm)
        prior_sd.append(np.array(node.marginal_priors)[:, i, 0].std())
        mc = np.array(node.mean_copies)
        burn = int(len(mc) * 0.1)
        post_sd.append(mc[burn:, i, 0].std())
        labels.append(f"m = {m}\n{g}[{i}]")

    prior_e = np.array(prior_e); prior_sd = np.array(prior_sd)
    post_e = np.array(post_e); post_sd = np.array(post_sd)
    xpos = np.arange(len(ms))
    off = 0.12

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0, color="red", lw=1.5, zorder=1, label="Truth (error = 0)")
    ax.errorbar(xpos - off, prior_e, yerr=2 * prior_sd, fmt="o",
                color="tab:blue", capsize=4, zorder=3,
                label="Prior marginal mean  ±2σ")
    ax.errorbar(xpos + off, post_e, yerr=2 * post_sd, fmt="s",
                color="tab:green", capsize=4, zorder=3,
                label="Posterior mean  ±2σ")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_xlabel("m  (observations per tri-gram) — representative gram shown at each m")
    ax.set_ylabel("Estimate − True   (0 = perfect)")
    ax.set_title(
        "Learning phase for tri-grams: approximation error vs m\n"
        "median-error gram per m; good estimator → interval covers 0, "
        "width shrinks with m",
        fontsize=10,
    )
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()
