import numpy as np
import pandas as pd
import distinctipy
import matplotlib.pyplot as plt
import copy
import umap
import time

from os.path import join
from lifelines import statistics
from lifelines.utils import concordance_index
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

colors =[
    'tab:green',
    'tab:blue',
    'tab:orange',
    'tab:red',
    'tab:purple',
    'tab:brown',
    'tab:olive',
    'tab:pink',
    'tab:gray',
    'tab:cyan']

fontsize = 15

def plot_cluster_traj(x, groups, n_group, labels, save_path):
    n_sample, seq_len, n_var = x.shape
    n_row, n_col = 9, 5
    step_var = n_var // n_col

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*2, n_row*2))
    l_list, group_labels = [], []
    for i in range(n_row):
        for j in range(n_col):
            obs_idx = i * n_col + j
            if obs_idx < len(labels):
                y_min, y_max = 1, -1
                for c in range(n_group):
                    group_vals = copy.deepcopy(x[groups==c, :, obs_idx])
                    if labels[obs_idx] in ['Age', 'Gender']:
                        group_vals = np.tile(group_vals[:, [0]], (1, group_vals.shape[1]))
                    else:
                        group_vals[group_vals==0] = np.nan
                    group_avg = np.nanmean(group_vals, axis=0)
                    group_std = np.nanstd(group_vals, axis=0)
                    l, = axes[i, j].plot(group_avg, c=colors[c])
                    if i == 0 and j == 0:  # Only add the legend item once
                        l_list.append(l)
                        group_labels.append('S{:} N={:}'.format(c+1, np.sum(groups==c)))
                    y_min, y_max = min(y_min, np.amin(group_avg)), max(y_max, np.amax(group_avg))
                axes[i, j].set_title(labels[obs_idx])
                axes[i, j].set_ylim(min(-1, y_min) - 0.02, max(1, y_max) + 0.02)
                axes[i, j].set_xticks(np.arange(0, seq_len + 1, 4))
                axes[i, j].set_xticklabels(np.arange(0, seq_len + 1, 4) * 4 - 24)

    fig.text(0.5, 0.04, 'Hours relative to sepsis onset', ha='center', fontsize=fontsize)
    fig.text(0.04, 0.5, 'Average of normalised metrics', va='center', rotation='vertical', fontsize=fontsize)
    fig.legend(handles=l_list, labels=group_labels, bbox_to_anchor=(0.5, 1.0), loc='upper center', ncol=min(n_group, 5), )
    fig.tight_layout(rect=[0.05, 0.05, 1, 0.96])
    plt.savefig(join(save_path, 'cluster_traj.png'), bbox_inches='tight')
    plt.close()

    selected_var = [
        'Age', 'Shock Index', 'Platelets Count', 'Heart Rate', 'PaCO2', 'Calcium', 
        'SOFA', 'GCS', 'Arterial Lactate', 'BUN', 'Total Input', 'Mechanical Ventilation'
    ]
    n_row, n_col = 2, 6
    l_list, group_labels = [], []
    fig, axes = plt.subplots(n_row, n_col, figsize=(11, 4))
    for i, label in enumerate(selected_var):
        y_min, y_max = 1, -1
        obs_idx = labels.index(label)
        for c in range(n_group):
            group_vals = copy.deepcopy(x[groups==c, :, obs_idx])
            if labels[obs_idx] in ['Age', 'Gender']:
                group_vals = np.tile(group_vals[:, [0]], (1, group_vals.shape[1]))
            else:
                group_vals[group_vals==0] = np.nan
            group_avg = np.nanmean(group_vals, axis=0)
            group_std = np.nanstd(group_vals, axis=0)
            l, = axes[int(i//n_col), int(i%n_col)].plot(group_avg, c=colors[c])
            if i == 0:
                l_list.append(l)
                group_labels.append('S{:}'.format(c+1))
            y_min, y_max = min(y_min, np.amin(group_avg)), max(y_max, np.amax(group_avg))
        axes[int(i//n_col), int(i%n_col)].set_title(label)
        axes[int(i//n_col), int(i%n_col)].set_ylim(min(-1, y_min) - 0.02, max(1, y_max) + 0.02)
        axes[int(i//n_col), int(i%n_col)].set_xticks(np.arange(0, seq_len + 1, 4))
        axes[int(i//n_col), int(i%n_col)].set_xticklabels(np.arange(0, seq_len + 1, 4) * 4 - 24) # the 24h before and 56h after onset time

    fig.text(0.5, 0.04, 'Hours relative to sepsis onset', ha='center', fontsize=fontsize)
    fig.text(0.04, 0.5, 'Average of normalised metrics', va='center', rotation='vertical', fontsize=fontsize)
    fig.legend(handles=l_list, labels=group_labels, bbox_to_anchor=(0.5, 1.0), loc='upper center', ncol=len(selected_var), )
    fig.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.savefig(join(save_path, 'cluster_traj_select.png'), bbox_inches='tight')
    plt.close()

def get_logrank_p(times, events, assignments):
    logrank_results = statistics.multivariate_logrank_test(times, assignments, events)
    return logrank_results.p_value

def get_km_curve(times, events, clip_time=90):
    unique_times = np.asarray(list(set(times)))
    sorted_unique_times = np.sort(unique_times)
    S_list = [1.]
    time_list = [0.]
    censor_list = [False]
    at_risk_list = [len(times)]
    live_at_the_start = len(times)
    S_t = 1.
    start_time = 0
    RMST = 0.
    for i in range(len(sorted_unique_times)):
        end_time = sorted_unique_times[i]
        event_num = np.sum(events[times==end_time])
        at_risk_list.append(live_at_the_start)
        live_at_the_start = np.sum(times >= end_time)
        if end_time <= clip_time:
            RMST += (S_t * (end_time - start_time))
        S_list.append(S_t)
        S_t *= (1. - event_num/live_at_the_start)
        S_list.append(S_t)
        time_list.append(end_time)
        time_list.append(end_time)
        censor_list.append(0 in events[times==end_time])
        censor_list.append(0 in events[times==end_time])
        at_risk_list.append(live_at_the_start)
        start_time = end_time
    if np.amax(times) < clip_time:
        RMST += (S_t * (60 - end_time))
    return S_list, time_list, censor_list, at_risk_list, RMST

def plot_km_curve(times, events, groups, n_group, save_path, risk=None):
    fig, ax = plt.subplots(figsize=(4, 5))

    rmst_list = []
    for i in range(n_group):
        if np.sum(groups == i) == 0:
            print('zero-size group found!')
            continue
        S_list, time_list, censor_list, at_risk_list, RMST = get_km_curve(
            times[groups==i], 
            events[groups==i], 
            90
        )
        rmst_list.append((RMST))
        ax.plot(time_list[:-1], S_list[:-1], c=colors[i], label='S'+str(i+1))

    if n_group > 1:
        logrank_p = get_logrank_p(times, events, groups)
        p_string = 'Log-rank p = {:.5f}'.format(logrank_p) if logrank_p >= 1e-4 else 'Log-rank p < 0.0001'
        if risk is not None:
            ci = concordance_index(times, 1 - risk, events)
            p_string += '\nC-index = {:.3f}'.format(ci)
        ax.text(43, 0.55, p_string)
        

    ax.grid()
    ax.legend(loc='lower left')
    ax.set_ylim(0.5, 1.02)
    ax.set_xlim(0, 90)
    ax.set_xlabel('Days', fontsize=fontsize)
    ax.set_ylabel('Survival probability', fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(join(save_path, 'km_curv.png'))

def group_compare_risk(risk, groups, n_group, save_path):
    fig, ax = plt.subplots(figsize=(4, 3))
    risk_list = [risk[groups == c] for c in range(n_group)]
    ax.boxplot(risk_list, labels=['S'+str(c+1) for c in range(n_group)])
    ax.set_ylabel('Predicted risk', fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(join(save_path, 'group_risks.png'))
    plt.close()

def plot_elbow(n_clusters_dist, k_list, save_path):
    fig, ax = plt.subplots(figsize=(5, 2))
    
    l1, = ax.plot(k_list, n_clusters_dist, c=colors[0], marker="D")
    ax.set_xlabel('K', fontsize=fontsize)
    ax.set_ylabel('Inertia', fontsize=fontsize)
    ax.tick_params(axis='y', labelcolor=colors[0])

    differences = [n_clusters_dist[i] - n_clusters_dist[i + 1] for i in range(len(n_clusters_dist) - 1)]
    ax2 = ax.twinx()
    l2, = ax2.plot(k_list[1:], differences, marker="o", c=colors[1])
    ax2.set_ylabel('Δ inertia', fontsize=fontsize)
    ax2.tick_params(axis='y', labelcolor=colors[1])

    l3, = ax.plot([4, 4], [np.amax(n_clusters_dist), np.amin(n_clusters_dist)], linestyle='--', c='black')

    plt.legend(
        handles=[l3], 
        labels=['Selected K'],
        loc='best'
    )
    plt.tight_layout()
    plt.savefig(join(save_path, 'elbow_with_diff.png'))
    plt.close()

def viz_2d(data, centroids, groups, n_cluster, risk, mode, seed, save_path):
    start_time = time.time()
    if mode == 'TSNE':
        reducer = TSNE(
            n_components=2, 
            early_exaggeration=12,
            learning_rate=100, 
            n_iter=1000,
            init='pca', 
            perplexity=50, 
            n_jobs=None,
            random_state=seed
        )
    elif mode == 'UMAP':
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=20,
            min_dist=0.,
            random_state=seed,
            n_jobs=1
        )
    else:
        assert False, 'unknown dimension reducer'
    data_2d = reducer.fit_transform(np.r_[data, centroids])
    data_2d, cent_2d = np.split(data_2d, [data.shape[0]], axis=0)
    print(mode + ' time: {:.1f}'.format(time.time() - start_time))

    fig, ax = plt.subplots(figsize=(6, 6))
    for c in range(n_cluster):
        ax.scatter(data_2d[groups==c, 0], data_2d[groups==c, 1], color=colors[c],
                   marker='.', label='S{:}'.format(c+1), s=10, alpha=0.7)
        ax.scatter(cent_2d[c, 0], cent_2d[c, 1], color=colors[c][4:], marker='X', edgecolors='black', label='C'+str(c+1))
    ax.legend()
    ax.set_xlabel('Axis 1 of the latent space with ' + mode, fontsize=fontsize)
    ax.set_ylabel('Axis 2 of the latent space with ' + mode, fontsize=fontsize)

    dbi = davies_bouldin_score(data_2d, groups)
    vrc = calinski_harabasz_score(data_2d, groups)
    text = 'DBI = {:.3f}\nVRC = {:.3f}'.format(dbi, vrc)
    for c in range(n_cluster):
        text += '\nS{:}(N={:})'.format(c+1, np.sum(groups==c))
    ax.text(np.amin(data_2d[:, 0]), np.amin(data_2d[:, 1]), text)

    plt.tight_layout()
    plt.savefig(join(save_path, 'viz_2d_' + mode + '.png'))
    plt.close()

    # c-index of 5 repeated experiments: [0.747, 0.753, 0.748, 0.752, 0.749], 0.750±0.002