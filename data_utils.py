import os
import pandas as pd
import numpy as np
import umap
import requests
import time
import progressbar
import copy

from os.path import join
from scipy.stats import zscore, kruskal
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def load_data(data_path, seed):
    MIMICtable = pd.read_csv(join(data_path, 'sepsis_final_data_RAW_withTimes_with_SurvivalDays.csv'), header=0)
    MIMICtable['traj'] -= 1
    raw_cols = list(MIMICtable)
    reformat = MIMICtable.values.copy()

    cols2norm = [
        'gender', 'mechvent', 'age', 'Weight_kg', 'Magnesium',
        'SOFA', 'SIRS', 'GCS', 'HR', 'FiO2_1', 'Potassium', 'Arterial_BE',
        'SysBP', 'MeanBP', 'DiaBP', 'RR', 'Temp_C', 'SGPT',
        'Sodium', 'Chloride', 'Glucose', 'Calcium', 'Hb', 
        'WBC_count', 'Platelets_count', 'PTT', 'PT', 'Arterial_pH', 
        'paO2', 'paCO2', 'HCO3', 'Arterial_lactate',  'Shock_Index', 
        'PaO2_FiO2', 'cumulated_balance'
    ] 
    cols2log = [
        'SpO2', 'BUN', 'Creatinine', 'SGOT', 'Total_bili', 'INR',
        'output_total', 'output_4hourly', 'input_total'
    ]
    scale_cols = cols2norm + cols2log
    cols2norm = np.array([raw_cols.index(item) for item in cols2norm])
    cols2log = np.array([raw_cols.index(item) for item in cols2log])
    scaleMIMIC = np.concatenate([zscore(reformat[:, cols2norm], ddof=1),
                                 zscore(np.log(0.1 + reformat[:, cols2log]), ddof=1)], axis=1)

    meta_cols = ['traj',  'step']
    obse_cols = [ 
    'GCS', 'SOFA', 'SIRS', 'HR', 'Magnesium', 
    'FiO2_1', 'Potassium', 'Arterial_BE', 'SysBP', 'MeanBP', 
    'DiaBP', 'RR', 'Temp_C', 'SGPT', 'Sodium', 
    'Chloride', 'Glucose', 'Calcium', 'Hb', 'WBC_count', 
    'Platelets_count', 'PTT', 'PT', 'Arterial_pH', 'paO2', 
    'paCO2', 'HCO3', 'Arterial_lactate',  'Shock_Index','SpO2', 
    'BUN', 'Creatinine', 'SGOT', 'Total_bili', 'INR', 
    'PaO2_FiO2', 'cumulated_balance', 'input_total', 'output_total','gender', 
    'age', 'Weight_kg', 'mechvent',
    ] 
    surv_cols = ['mortality_90d', 'survival_days']

    meta_ids = np.array([raw_cols.index(item) for item in meta_cols])
    obse_ids = np.array([scale_cols.index(item) for item in obse_cols])
    surv_ids = np.array([raw_cols.index(item) for item in surv_cols])

    meta = reformat[:, meta_ids]
    cond = scaleMIMIC[:, cond_ids]
    obse = scaleMIMIC[:, obse_ids]
    surv = reformat[:, surv_ids]

    # cut-off survival days at 90 days and remove negative value
    surv_d = surv[:, 1]
    surv_d[surv_d < 0] = 0
    surv_d[np.isnan(surv_d)] = -1
    surv[:, 1] = surv_d

    # assign the same surv label to all obse of a patient
    pids = meta[:, 0]
    unique_pids = np.unique(pids)
    for i in unique_pids:
        surv[pids==i] = surv[pids==i][-1]

    # add train/valid/test to the column 'group' in category 'meta'
    group = np.zeros((surv.shape[0]))
    temp_df = pd.DataFrame(np.c_[meta[:, 0], surv[:, 0] ], columns=['traj', 'label'])
    train_ids, valid_ids, test_ids = stratified_sequence_split(temp_df, 0.25, 0.2, seed)
    group[train_ids] = 0
    group[valid_ids] = 1
    group[test_ids] = 2
    meta_cols.append('group')

    obse_cols = check_word(obse_cols)

    columns = [(cat, col) for cat, cols in zip(['meta', 'surv', 'obse'], [meta_cols, surv_cols, obse_cols]) for col in cols]
    multi_index = pd.MultiIndex.from_tuples(columns, names=['Category', 'Name'])
    df = pd.DataFrame(np.c_[meta, group, action, surv, obse], columns=multi_index)

    for i, set_name in enumerate(['train', 'valid', 'test']):
        sub_df = df[df[('meta', 'group')]==i]
        sequence_labels = sub_df.groupby([('meta', 'traj')]).apply(lambda x: x[('surv', 'mortality_90d')].iloc[-1])
        labels = sequence_labels.to_numpy(dtype=int)
        label_distribution = np.bincount(labels)
        print(set_name, "label distribution:", label_distribution, np.mean(labels))

    return df, MIMICtable

def check_word(words):
    replace_dict = {'FiO2_1': 'FiO2', 'Arterial_BE': 'Arterial Base Excess', 'SysBP':'Systolic Blood Pressure',
    'MeanBP': 'Mean Blood Pressure', 'DiaBP': 'Diastolic Blood Pressure', 'RR': 'Respiratory rate', 'HR': 'Heart Rate',
    'Temp_C': 'Temperature (℃)', 'SGPT': 'Serum Glutamic\nPyruvic Transaminase', 'Hb': 'Hemoglobin',
    'WBC_count': 'White Blood Cell', 'PTT': 'Partial Thromboplastin Time', 'PT': 'Prothrombin Time',
    'SGOT': 'Serum Glutamic\nOxaloacetic Transaminase', 'Total_bili': 'Total Bilirubin', 'INR': 'International Normalized Ratio',
    'cumulated_balance': 'Cumulated Balance', 'Weight_kg':'Weight (kg)', 'mechvent': 'Mechanical Ventilation',
    'input_total': 'Total Input', 'output_total': 'Total Output', 'age': 'Age', 'gender': 'Gender',
    'PaO2_FiO2': 'PaO2/FiO2 Ratio', 'Shock_Index': 'Shock Index', 'Arterial_lactate': 'Arterial Lactate',
    'Platelets_count': 'Platelets Count', 'paCO2':'PaCO2', 'paO2': 'PaO2', 'Arterial_pH': 'Arterial Potential of Hydrogen'
    }
    new_words = [replace_dict[word] if word in replace_dict.keys() else word for word in words]
    return new_words

def stratified_sequence_split(df, valid_rate, test_rate, seed):
    # Extract sequence IDs and labels
    traj_ids = df['traj'].unique()
    # Assuming one label per sequence, get the last occurrence of each sequence for its label
    labels = df.groupby('traj')['label'].last().to_numpy()
    
    # First split to separate out the test set, stratified by label
    rest_ids, test_ids = train_test_split(traj_ids,
                                          test_size=test_rate,
                                          stratify=labels,
                                          random_state=seed)

    # Get labels for the remaining sequences
    rest_labels = df[df['traj'].isin(rest_ids)].groupby('traj')['label'].last().to_numpy()

    # Split the remaining sequences into training and validation, stratified by label
    train_ids, valid_ids = train_test_split(rest_ids,
                                            test_size=valid_rate,
                                            stratify=rest_labels,
                                            random_state=seed)

    # Convert IDs back to row indices for each set
    train_indices = df[df['traj'].isin(train_ids)].index.to_numpy()
    valid_indices = df[df['traj'].isin(valid_ids)].index.to_numpy()
    test_indices = df[df['traj'].isin(test_ids)].index.to_numpy()

    return train_indices, valid_indices, test_indices

def dim_reduction(data, mode='tsne', random_state=0, reducer=None):
    if reducer is None:
        if mode == 'tsne':
            reducer = TSNE(
                n_components=2, 
                early_exaggeration=12,
                learning_rate=100, 
                n_iter=1000,
                init='pca', 
                perplexity=50, 
                n_jobs=None,
                random_state=random_state
            )
            data_2d = reducer.fit_transform(data)
            print(mode + ' KL_div={:.2f}, n_iter={:}'.format(reducer.kl_divergence_, reducer.n_iter_))
        elif mode == 'umap':
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=500,
                min_dist=0.,
                random_state=random_state,
                n_jobs=1
            )
            data_2d = reducer.fit_transform(data)
    else:
        data_2d = reducer.transform(data)
    return data_2d, reducer

def create_numpy_data(df, max_seq_len, surv_time=False):
    unique_pids = np.unique(df[('meta', 'traj')].to_numpy())
    # create the test data matrix with zero padding
    X = []
    y = []
    seq_len = []
    t = []
    for pid in unique_pids:
        sequence_data = df[df[('meta', 'traj')] == pid]
        X_seq = sequence_data.loc[:, ('obse', slice(None))].to_numpy()
        if len(X_seq) > max_seq_len:
            X_seq = X_seq[:max_seq_len]
        X_padded =  np.pad(X_seq, ((0, max_seq_len - len(X_seq)), (0, 0)), 'constant', constant_values=0)
        X.append(X_padded)
        y_seq = sequence_data[('surv', 'mortality_90d')].iloc[-1]
        y.append(y_seq)
        seq_len.append(len(X_seq))
        t.append(df[df[('meta', 'traj')] == pid][('surv', 'survival_days')].to_list()[-1])
    X = np.array(X)
    y = np.array(y)
    t = np.array(t)
    seq_len = np.array(seq_len)
    if surv_time:
        return X, y, seq_len, t
    else:
        return X, y, seq_len

def stratify_patients(predicted_risk, n_clusters):
    # Number of samples
    n_samples = predicted_risk.shape[0]
    
    # Sort the predicted risks and get the sorted indices
    sorted_indices = np.argsort(predicted_risk)
    
    # Calculate the number of samples per cluster
    samples_per_cluster = np.full(n_clusters, n_samples // n_clusters)
    samples_per_cluster[:n_samples % n_clusters] += 1
    
    # Assign cluster labels
    clusters = np.zeros(n_samples, dtype=int)
    start = 0
    for i, size in enumerate(samples_per_cluster):
        clusters[sorted_indices[start:start + size]] = i
        start += size
    
    return clusters

def rename_cluster(clus, risk, n_cluster, return_ids=False):
    clus_means = []
    for i in range(n_cluster):
        if np.sum(clus==i) > 0:
            clus_means.append(np.mean(risk[clus==i]))
        else:
            clus_means.append(0)
    ids = np.argsort(clus_means)
    clus_temp = np.zeros_like(clus)
    for i in range(n_cluster):
        clus_temp[clus==ids[i]] = i
    if return_ids:
        return clus_temp, ids
    else:
        return clus_temp
