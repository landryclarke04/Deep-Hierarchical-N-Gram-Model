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
    """Estimator errors for every component of every fitted node.

    Returns (prior_err, post_err, mle_err) as flat arrays of (estimate - true).
    Prior and posterior errors span the whole hierarchy; the MLE error is only
    defined on data-bearing tri-grams (no sharing -> nothing to estimate at an
    internal or data-less node), so its array is shorter.
    """
    prior_err = []
    post_err = []
    mle_err = []
    for gram, node in model.nodes.items():
        tm = node.get_true_mean()
        pm = node.get_marginal_prior_mean()
        em = node.get_est_mean()
        if tm is None or pm is None or em is None:
            continue
        tmf = tm.flatten(); pm = pm.flatten(); em = em.flatten()
        prior_err.extend((pm - tmf).tolist())
        post_err.extend((em - tmf).tolist())
        mle = node.get_mle_mean()
        if mle is not None:
            mle_err.extend((mle.flatten() - tmf).tolist())
    return np.array(prior_err), np.array(post_err), np.array(mle_err)


def make_learning_error_plot(model, n_show=18):
    """Two-panel learning figure.

    Left  : per-gram dumbbell of error (prior -> posterior), truth at 0, with
            the MLE baseline (own data only) shown where it is defined.
    Right : ECDF of |error| (prior vs posterior over the whole hierarchy; MLE
            over the data-bearing tri-grams where it exists).
    """
    # ----- aggregate over every component -----
    all_prior_err, all_post_err, all_mle_err = _collect_errors(model)

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
        # MLE baseline (own sample mean, no sharing); NaN where undefined.
        mle = node.get_mle_mean()
        se = node.get_mle_se()
        me = mle.flatten()[i] - tm if mle is not None else np.nan
        mse = se.flatten()[i] if se is not None else (
            0.0 if mle is not None else np.nan)
        rows.append((f"{g}[{i}]", pe, po, mp.std(), mc.std(), me, mse))
    # largest prior error at the top
    rows.sort(key=lambda r: abs(r[1]))
    labels = [r[0] for r in rows]
    prior_e = np.array([r[1] for r in rows])
    post_e = np.array([r[2] for r in rows])
    prior_sd = np.array([r[3] for r in rows])
    post_sd = np.array([r[4] for r in rows])
    mle_e = np.array([r[5] for r in rows])
    mle_sd = np.array([r[6] for r in rows])
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
    mle_mask = ~np.isnan(mle_e)
    if mle_mask.any():
        axL.errorbar(mle_e[mle_mask], y[mle_mask], xerr=2 * mle_sd[mle_mask],
                     fmt="D", color="tab:red", capsize=3, markersize=6,
                     zorder=4, label="MLE, own data only  ±2·SE")
    axL.set_yticks(y)
    axL.set_yticklabels(labels, fontsize=8)
    axL.set_xlabel("Estimate − True   (0 = perfect)")
    axL.set_title("Per-gram error ±2σ  (bar crossing 0 ⇒ interval covers truth)",
                  fontsize=10)
    axL.legend(fontsize=8, loc="lower right")

    # ===== Panel B: FAIR |error| ECDF on the common support =====
    # The MLE only exists on data-bearing tri-grams, so to compare estimators
    # head-to-head we score all three on exactly that subset (otherwise the
    # MLE looks artificially strong -- it is only ever asked the easy,
    # data-rich questions). The whole-hierarchy "posterior beats prior" story
    # lives in the left panel and the suptitle.
    def ecdf(a):
        s = np.sort(a)
        return s, np.arange(1, len(s) + 1) / len(s)

    dp = []; dq = []; dr = []   # prior / posterior / mle on data tri-grams
    for g in model.nodes:
        if len(g) != 3:
            continue
        node = model.nodes[g]
        tm = node.get_true_mean()
        mle = node.get_mle_mean()
        if tm is None or mle is None:
            continue
        tmf = tm.flatten()
        dp.extend(np.abs(node.get_marginal_prior_mean().flatten() - tmf))
        dq.extend(np.abs(node.get_est_mean().flatten() - tmf))
        dr.extend(np.abs(mle.flatten() - tmf))
    dp = np.array(dp); dq = np.array(dq); dr = np.array(dr)

    xp, yp = ecdf(dp)
    xq, yq = ecdf(dq)
    xr, yr = ecdf(dr)
    axR.plot(xp, yp, color="tab:blue", lw=2,
             label=f"Prior  (median {np.median(dp):.2f})")
    axR.plot(xq, yq, color="tab:green", lw=2,
             label=f"Posterior — shares info  (median {np.median(dq):.2f})")
    axR.plot(xr, yr, color="tab:red", lw=2, ls="-.",
             label=f"MLE — no sharing  (median {np.median(dr):.2f})")
    axR.set_xlabel("|Estimate - True|")
    axR.set_ylabel("Fraction of components <= x")
    axR.set_title(f"Fair comparison on data tri-grams (N={len(dr)} comps)")
    axR.legend(fontsize=8, loc="lower right")

    beats = float(np.mean(np.abs(all_post_err) < np.abs(all_prior_err))) * 100
    fig.suptitle(
        "Effectiveness of learning: posterior beats prior for "
        f"{beats:.0f}% of components",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


def make_error_vs_m_plot(model):
    """Error vs. sample size for the posterior mean vs. the MLE.

    For every tri-gram we compute the RMSE (over that gram's mean components)
    of two estimators:
      - MLE (own sample mean): NO hierarchical sharing; undefined at m=0.
      - Gibbs posterior mean : conditions on the data AND shares across the
                               hierarchy -> lower error at small m, converging
                               to the MLE as m grows.

    Each curve is the median RMSE across tri-grams with an inter-quartile
    band (robust to the right-skew of RMSE; never dips below 0). The y-axis
    is log-scaled. The MLE curve has no point at m=0 (no data -> undefined).
    """
    from collections import defaultdict

    def rmse(est, true):
        return float(np.sqrt(np.mean((est.flatten() - true.flatten()) ** 2)))

    # (attribute getter, color, line style, legend label)
    curves = {
        "mle":       ("get_mle_mean", "tab:red",   "D-.",
                      "MLE — own data only (no sharing)"),
        "posterior": ("get_est_mean", "tab:green", "s-",
                      "Posterior mean — shares info"),
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
        # a curve may be undefined for some m (e.g. MLE at m=0): plot only
        # the ticks where it has data.
        idx = [j for j, m in enumerate(ms) if by_m[k][m]]
        if not idx:
            continue
        xk = np.array([xpos[j] for j in idx])
        med = np.array([np.median(by_m[k][ms[j]]) for j in idx])
        q25 = np.array([np.percentile(by_m[k][ms[j]], 25) for j in idx])
        q75 = np.array([np.percentile(by_m[k][ms[j]], 75) for j in idx])
        ax.fill_between(xk, q25, q75, color=color, alpha=0.15)
        ax.plot(xk, med, fmt, color=color, lw=2, label=label)

    ax.set_yscale("log")
    ax.set_xticks(xpos)
    ax.set_xticklabels([str(m) for m in ms])
    ax.set_xlabel("m  (observations per tri-gram)")
    ax.set_ylabel("RMSE per tri-gram  (median, IQR band; log scale)")
    ax.set_title("Posterior mean vs. MLE: error vs. sample size m")
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
    mle_x = []; mle_e = []; mle_sd = []   # MLE only where it is defined (m>0)
    for j, m in enumerate(ms):
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

        # MLE baseline (own sample mean, no sharing); absent at m=0.
        mle = node.get_mle_mean()
        se = node.get_mle_se()
        if mle is not None:
            mle_x.append(j)
            mle_e.append(mle.flatten()[i] - tm)
            mle_sd.append(se.flatten()[i] if se is not None else 0.0)

    prior_e = np.array(prior_e); prior_sd = np.array(prior_sd)
    post_e = np.array(post_e); post_sd = np.array(post_sd)
    mle_x = np.array(mle_x); mle_e = np.array(mle_e); mle_sd = np.array(mle_sd)
    xpos = np.arange(len(ms))
    off = 0.16

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0, color="red", lw=1.5, zorder=1, label="Truth (error = 0)")
    ax.errorbar(xpos - off, prior_e, yerr=2 * prior_sd, fmt="o",
                color="tab:blue", capsize=4, zorder=3,
                label="Prior marginal mean  ±2σ")
    if len(mle_x):
        ax.errorbar(mle_x, mle_e, yerr=2 * mle_sd, fmt="D",
                    color="tab:red", capsize=4, zorder=3,
                    label="MLE, own data only  ±2·SE")
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


def make_posterior_predictive_check(model, n_new=40):
    """Predictive posterior mean compared to new data, per representative tri-gram.

    One tick per value of m, using the same median-error representative
    tri-gram+component as the learning plot. At each tick we compare:

      - Posterior predictive for a NEW observation y: draw y ~ N(mu, true_var)
        across the Gibbs posterior samples mu of the mean, then summarize by
        mean +/- 2 sigma (this folds in both posterior uncertainty about the
        mean AND the observation noise true_var).
      - Genuinely NEW data the model never saw: fresh draws
        y ~ N(true_mean, true_var) from the true data-generating process.

    A well-fit model's predictive band should cover the held-out data, and the
    predictive mean should track the truth -- increasingly so as m grows.
    """
    from collections import defaultdict

    # representative (gram, component) per m = median |posterior-mean error|
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

    labels = []
    pred_mean = []; pred_sd = []; true_vals = []; new_pts = []
    mle_x = []; mle_pred = []            # MLE plug-in prediction (y-bar); m>0 only
    for j, m in enumerate(ms):
        lst = sorted(cand[m], key=lambda r: r[0])
        _, g, i = lst[len(lst) // 2]
        node = model.nodes[g]
        burn = int(len(node.mean_copies) * 0.1)

        # posterior predictive draws of a new observation y
        yy = np.array([
            node.sample_MVN(mu, node.true_var).flatten()[i]
            for mu in node.mean_copies[burn:]
        ])
        pred_mean.append(yy.mean())
        pred_sd.append(yy.std())

        # genuinely new data from the TRUE parameters
        new_pts.append([
            node.sample_MVN(node.true_mean, node.true_var).flatten()[i]
            for _ in range(n_new)
        ])

        true_vals.append(node.get_true_mean().flatten()[i])
        labels.append(f"m = {m}\n{g}[{i}]")

        # MLE plug-in prediction of new data: center at the sample mean y-bar
        # (no sharing). Undefined at m=0.
        mle = node.get_mle_mean()
        if mle is not None:
            mle_x.append(j)
            mle_pred.append(mle.flatten()[i])

    xpos = np.arange(len(ms))
    pred_mean = np.array(pred_mean); pred_sd = np.array(pred_sd)
    mle_x = np.array(mle_x); mle_pred = np.array(mle_pred)

    fig, ax = plt.subplots(figsize=(10, 6))

    # held-out new data
    for j, (xp, pts) in enumerate(zip(xpos, new_pts)):
        jitter = np.random.uniform(-0.12, 0.12, len(pts))
        ax.scatter(np.full(len(pts), xp) + jitter, pts, marker="o", s=18,
                   color="red", alpha=0.5, zorder=2,
                   label="New data from true params" if j == 0 else "")

    # MLE plug-in prediction (sample mean, no sharing)
    if len(mle_x):
        ax.scatter(mle_x, mle_pred, marker="D", s=70, color="tab:purple",
                   zorder=5, label="MLE prediction ȳ (no sharing)")

    # posterior predictive mean +/- 2 sigma
    ax.errorbar(xpos, pred_mean, yerr=2 * pred_sd, fmt="s", color="tab:green",
                capsize=5, markersize=8, lw=2, zorder=4,
                label="Posterior predictive mean  ±2σ")

    # true mean, for reference
    ax.scatter(xpos, true_vals, marker="_", s=400, color="black", zorder=3,
               label="True mean")

    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_xlabel("m  (observations per tri-gram) — representative gram shown at each m")
    ax.set_ylabel("Value of a new observation y")
    ax.set_title(
        "Predictive posterior mean compared to new data\n"
        "predictive band should cover the held-out samples",
        fontsize=10,
    )
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()
