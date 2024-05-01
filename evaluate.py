import numpy as np
import pandas as pd
import os
from typing import Union, List

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, average_precision_score
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics.cluster import contingency_matrix
import viz_utils

def compute_supervised_scores(y_true: np.ndarray, y_pred: np.ndarray, thresh: float):
    auroc = roc_auc_score(y_true, y_pred)
    
    labels_true = y_true
    labels_pred = (y_pred > thresh).astype(float)

    # Compute F1
    f1 = f1_score(labels_true, labels_pred)

    # Compute Recall
    rec = recall_score(labels_true, labels_pred)

    # Compute Precision
    prec = precision_score(labels_true, labels_pred)

    # Compute ARI
    ari = adjusted_rand_score(labels_true, labels_pred)

    # Compute NMI
    nmi = normalized_mutual_info_score(labels_true, labels_pred)

    # Return Dictionary
    scores_dic = {
        "ROC-AUC": auroc,
        "F1": f1,
        "Recall": rec,
        "Precision": prec,
        "ARI": ari,
        "NMI": nmi
    }

    return scores_dic

def compute_cluster_performance(X, clus_pred, y_true):
    # If not converted to categorical, then convert
    if len(clus_pred.shape) == 2:
        clus_pred = np.argmax(clus_pred, axis=1)

    # Compute the same taking average over each feature dimension
    sil_avg, dbi_avg, vri_avg = 0, 0, 0

    for feat in range(X.shape[-1]):
        sil_avg += silhouette_score(X[:, :, feat], clus_pred, metric="euclidean")
        dbi_avg += davies_bouldin_score(X[:, :, feat], clus_pred)
        vri_avg += calinski_harabasz_score(X[:, :, feat], clus_pred)

    # Compute average factor
    num_feats = X.shape[-1]

    # Return Dictionary
    clus_perf_dic = {
        "DBI": dbi_avg / num_feats,
        "VRI": vri_avg / num_feats,
        "SIL": sil_avg / num_feats
    }

    return clus_perf_dic


def evaluate(x, y_true, y_pred, clus_pred, thresh, save_path, set_name):

    if thresh is None:
        # seach for thresh with the best f1-score
        result_list = []
        threshs = np.arange(0.1, 1., 0.1)
        f1_list = []
        for i, thresh in enumerate(threshs):
            scores = compute_supervised_scores(y_true, y_pred, thresh)
            result_list.append(scores)
            f1_list.append(scores['F1'])
        best_idx = np.argmax(f1_list)
        clf_metrics = result_list[best_idx]
        thresh = threshs[best_idx]
    else:
        clf_metrics = compute_supervised_scores(y_true, y_pred, thresh)
    clf_metrics['thresh'] = thresh

    if clus_pred is not None:
        clus_metrics = compute_cluster_performance(x, clus_pred=clus_pred, y_true=y_true)
    else:
        clus_metrics = {"DBI": 0.,"VRI": 0., "SIL": 0.}

    result_df = pd.DataFrame({**clf_metrics, **clus_metrics}, index=[0])
    result_df.to_csv(os.path.join(save_path, set_name + '_result.csv'))

    return thresh

