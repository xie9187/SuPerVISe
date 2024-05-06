import os
import time
import copy
import argparse
import platform
import numpy as np
import pandas as pd
import joblib
import progressbar
from os.path import join

from evaluate import evaluate
import data_utils
import viz_utils

sys = platform.system()
parser = argparse.ArgumentParser(description='Process some parameters.')
parser.add_argument('--random_seed', type=int, default=0)
parser.add_argument('--valid_rate', type=float, default=0.2)
parser.add_argument('--test_rate', type=float, default=0.2)
parser.add_argument('--b_size', type=int, default=32)
parser.add_argument('--max_seq_len', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--decay', type=float, default=1e-5)
parser.add_argument('--n_emb', type=int, default=32)
parser.add_argument('--n_hid', type=int, default=100)
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--n_clusters', type=int, default=4)
parser.add_argument('--learn_prior', type=bool, default=False)
parser.add_argument('--reduce_lr', type=bool, default=False)
parser.add_argument('--seq_att', type=bool, default=False)
parser.add_argument('--init_model', type=bool, default=False)
parser.add_argument('--feat_select', type=bool, default=False)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dim_red', type=str, default='TSNE')
parser.add_argument('--exp_name', type=str, default='analysis-uni_prior')
parser.add_argument('--model', type=str, default='seqVaGMM')
parser.add_argument('--data_path', default=r'D:\Data\MIMIC\mimic-iii_processed_data')
parser.add_argument('--data_path2', default=r'D:\Data\MIMIC\mimic-iv_processed_data_after2013')
parser.add_argument('--result_path', default=r'D:\Data\SepsisClustering')
parser.add_argument('--dataset', type=str, default='mimic-iii2iv')

args = parser.parse_args()

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

def SequantialVariationalGMM(df, save_path):
    import models.seqVaGMM as seqVaGMM
    import tensorflow as tf

    with tf.device('/gpu'):
        model = seqVaGMM.seqVaGMM(
            latent_dim=args.n_emb,
            hid_dim=args.n_hid,
            num_clusters=args.n_clusters,
            inp_shape=df.loc[:, ('obse', slice(None))].shape[1],
            max_seq_len=args.max_seq_len,
            survival=True,
            monte_carlo=1,
            sample_surv=False,
            learn_prior=args.learn_prior,
            seq_att=args.seq_att,
            seed=args.seed,
        )
    model_path = os.path.join(save_path, 'ckpt', 'model.ckpt')
    if os.path.exists(model_path + '.index'):
        model.load_weights(model_path).expect_partial()
        print('load model from: ' + model_path)
    else:
        model = seqVaGMM.train_model(
            df, 
            model, 
            lr=args.lr,
            decay=args.decay,
            b_size=args.b_size,
            num_epochs=args.n_epoch,
            reduce_lr=args.reduce_lr,
            save_path=save_path
        )
        model.save_weights(model_path)
    
    sub_df = df[df[('meta', 'group')] == 2]
    unique_pids = np.unique(sub_df[('meta', 'traj')].to_numpy())
    X, y, seq_len = data_utils.create_numpy_data(sub_df, args.max_seq_len)
    dec, z_sample, p_c_z, risk, p_z_c = seqVaGMM.test_model(model, X, np.expand_dims(y, axis=-1), seq_len, 64)
    risk = np.squeeze(risk)
    evaluate(x=X, y_true=y, y_pred=risk, clus_pred=p_c_z, thresh=0.5, save_path=save_path, set_name='test')

    pred_dict = {'traj': unique_pids, 'risk': risk, 'cluster': np.argmax(p_c_z, axis=-1)}
    pred_df = pd.DataFrame(pred_dict)
    pred_df.to_csv(join(save_path, 'preds.csv'))


def best_k(exp_path, df, plot=False):
    import re
    model_folders = sorted([x for x in os.listdir(exp_path) if 'cluster' in x])
    model_folders.sort(key=lambda s: int(re.findall(r'\d+', s)[0]))
    cols = [col[1] for col in df.columns if col[0]=='obse']
    X, y, seq_len, t = data_utils.create_numpy_data(df, args.max_seq_len, surv_time=True)
    df_list = []

    # search for the best number of clusters
    n_clusters_dist = []
    n_clusters = range(2, 11)
    for n_cluster in n_clusters:
        model_folder = 'cluster' + str(n_cluster)
        print(model_folder)
        seed_path = join(exp_path, model_folder)
        seed_folders = sorted([x for x in os.listdir(seed_path) if '.' not in x])
        dist_list = np.zeros((5))
        for seed, folder in enumerate(seed_folders):
            print(folder)
            pred_df = pd.read_csv(join(seed_path, folder, 'preds.csv'), index_col=0)
            risk = pred_df['risk'].to_numpy()
            clus = pred_df['cluster'].to_numpy()

            # calculate distance of samples to centroids
            for c in range(n_cluster):
                if np.mean(clus==c) == 0:
                    clus_total = 0
                else:
                    X_c = copy.deepcopy(X[clus==c, :, :])
                    X_c[X_c==0] = np.nan
                    centroids = np.nanmean(X_c, axis=0)
                    sd2c = np.nansum((X_c - centroids) ** 2, axis=1)
                    clus_total = np.sum(np.mean(sd2c, axis=1)) # average on clinical metrics and sum on trajectories 
                dist_list[seed] += clus_total

            if n_cluster == 4:
                # rename cluster with ascending mean predicted risk
                clus = data_utils.rename_cluster(clus, risk, n_cluster)

                # plot trajectory of clinical variabels
                viz_utils.plot_cluster_traj(X, clus, n_cluster, cols, join(seed_path, folder))
                
                # plot Kaplan-Meier curves
                d = np.zeros_like(t)
                d[np.logical_and(t > 0, t <= 90)] = 1 # dead
                d[np.logical_or(t < 0, t > 90)] = 0 # alive
                t[np.logical_or(t < 0, t > 90)] = 90
                viz_utils.plot_km_curve(t, d, clus, n_cluster, join(seed_path, folder), risk=risk)

                # plot difference of risks between groups
                viz_utils.group_compare_risk(risk, clus, n_cluster, join(seed_path, folder))

        dist = np.mean(dist_list)
        n_clusters_dist.append(dist)

        df = pd.concat([pd.read_csv(join(seed_path, folder, 'test_result.csv'), index_col=0) for folder in seed_folders])
        df_mean = df.mean(axis=0)
        df_std = df.std(axis=0) 
        val_str = df_mean.apply("{:.3f}".format) + '±' + df_std.apply("{:.3f}".format)
        result_df = pd.DataFrame([val_str], index=[model_folder])
        df_list.append(result_df)

    result_df = pd.concat(df_list)
    result_df.to_csv(join(exp_path, 'metrics.csv'))
    viz_utils.plot_elbow(n_clusters_dist, n_clusters, exp_path)

def viz_cluster(save_path, df):
    import models.seqVaGMM as seqVaGMM
    import tensorflow as tf

    with tf.device('/gpu'):
        model = seqVaGMM.seqVaGMM(
            latent_dim=args.n_emb,
            hid_dim=args.n_hid,
            num_clusters=args.n_clusters,
            inp_shape=df.loc[:, ('obse', slice(None))].shape[1],
            max_seq_len=args.max_seq_len,
            survival=True,
            monte_carlo=1,
            sample_surv=False,
            learn_prior=args.learn_prior,
            seq_att=args.seq_att,
            seed=args.seed,
        )
    model_path = os.path.join(save_path, 'ckpt', 'model.ckpt')
    model.load_weights(model_path).expect_partial()
    print('load model from: ' + model_path)

    X, y, seq_len = data_utils.create_numpy_data(df, args.max_seq_len)
    dec, z, p_c_z, risk, p_z_c = seqVaGMM.test_model(model, X, np.expand_dims(y, axis=-1), seq_len, 64)
    c_mean = model.c_mu.numpy()
    risk = np.squeeze(risk)
    clus = np.argmax(p_c_z, axis=-1)
    clus, sorted_ids = data_utils.rename_cluster(clus, risk, args.n_clusters, return_ids=True)
    c_mean = c_mean[sorted_ids]
    
    viz_utils.viz_2d(z, c_mean, clus, args.n_clusters, risk, args.dim_red, args.seed, save_path)

def explain(save_path, df):
    import models.seqVaGMM as seqVaGMM
    import tensorflow as tf
    import shap

    with tf.device('/gpu'):
        model = seqVaGMM.seqVaGMM(
            latent_dim=args.n_emb,
            hid_dim=args.n_hid,
            num_clusters=args.n_clusters,
            inp_shape=df.loc[:, ('obse', slice(None))].shape[1],
            max_seq_len=args.max_seq_len,
            survival=True,
            monte_carlo=1,
            sample_surv=False,
            learn_prior=args.learn_prior,
            seq_att=args.seq_att,
            seed=args.seed,
        )
    model_path = os.path.join(save_path, 'ckpt', 'model.ckpt')
    model.load_weights(model_path).expect_partial()
    print('load model from: ' + model_path)

    # only use data with seq_len==20
    train_df = df[df[('meta', 'group')] == 0]
    X_tr, y_tr, seq_len_tr = data_utils.create_numpy_data(train_df, args.max_seq_len)
    len_is_20 = seq_len_tr == 20
    X_tr = X_tr[len_is_20]
    y_tr = y_tr[len_is_20]
    seq_len_tr = seq_len_tr[len_is_20]

    test_df = df[df[('meta', 'group')] == 2]
    X_te, y_te, seq_len_te = data_utils.create_numpy_data(test_df, args.max_seq_len)
    len_is_20 = seq_len_te == 20
    X_te = X_te[len_is_20]
    y_te = y_te[len_is_20]
    seq_len_te = seq_len_te[len_is_20]

    # get the cluster prediction
    dec, z, p_c_z, risk, p_z_c = seqVaGMM.test_model(model, X_tr, np.expand_dims(y_tr, axis=-1), seq_len_tr, 64)
    clus_tr = np.argmax(p_c_z, axis=-1)
    dec, z, p_c_z, risk, p_z_c = seqVaGMM.test_model(model, X_te, np.expand_dims(y_te, axis=-1), seq_len_te, 64)
    clus_te = np.argmax(p_c_z, axis=-1)

    # wrap the risk prediction since SHAP needs the model to only output the variable to be explained
    def risk_wrapper(data):
        return seqVaGMM.risk_pred(model, data, np.ones((data.shape[0]), dtype=int) * 20, 64)

    # check wrapper and model performance AUC=0.791
    pred = risk_wrapper(X_te)
    from sklearn.metrics import roc_auc_score
    print('auc = ', roc_auc_score(y_te, pred))

    # use the centroid as the background sample and get the shapley value of the first 100 samples in each cluster
    feat_cols = [col[1] for col in df.columns if col[0] == 'obse']
    time_cols = ['T' + str(t) for t in range(args.max_seq_len)]
    feats_shap_dfs = []
    times_shap_dfs = []
    for c in range(args.n_clusters):
        feats_shap_df_path = join(save_path, 'feats_shap_S{:}.csv'.format(c+1))
        times_shap_df_path = join(save_path, 'times_shap_S{:}.csv'.format(c+1))

        if os.path.exists(feats_shap_df_path) and os.path.exists(times_shap_df_path):
            feats_shap_df = pd.read_csv(feats_shap_df_path, index_col=0)
            times_shap_df = pd.read_csv(times_shap_df_path, index_col=0)
        else:
            X_c = X_tr[clus_tr==c]
            X_bg = np.mean(X_c, axis=0)
            X_te_c = X_te[clus_te==c]
            samples_feats_shap, samples_times_shap = [], []
            for i in progressbar.progressbar(range(100)):
                X_sample = X_te_c[i]
                feats_shap = data_utils.shapley_values_time_series_groups(risk_wrapper, X_bg, X_sample, perturb_by_time=True)
                times_shap = data_utils.shapley_values_time_series_groups(risk_wrapper, X_bg, X_sample, perturb_by_time=False)
                samples_feats_shap.append(feats_shap)
                samples_times_shap.append(times_shap)

            feats_shap_df = pd.DataFrame(np.stack(samples_feats_shap), columns=feat_cols)
            feats_shap_df.to_csv(feats_shap_df_path)
            times_shap_df = pd.DataFrame(np.stack(samples_times_shap), columns=time_cols)
            times_shap_df.to_csv(times_shap_df_path)
        feats_shap_dfs.append(feats_shap_df)
        times_shap_dfs.append(times_shap_df)
    viz_utils.plot_shap(feats_shap_dfs, join(save_path, 'feats_shap.png'))
    viz_utils.plot_shap(times_shap_dfs, join(save_path, 'times_shap.png'), [str(t*4 - 20) + 'h' for t in range(20)])


if __name__ == '__main__':
    np.random.seed(args.seed)

    # load data
    if 'mimic-iii2iv' in args.dataset:
        df1, _ = data_utils.load_data(args.data_path, args.seed)
        df2, _ = data_utils.load_data(args.data_path2, args.seed)
        df1[df1[('meta', 'group')] == 2] = 1
        df2[('meta', 'group')] = 2
        df = pd.concat([df1, df2], axis=0)
    else:
        df, _ = data_utils.load_data(args.data_path, args.seed)

    if args.feat_select:
        train_val_df = df[df[('meta', 'group')] < 2]
        surv_feats = data_utils.feature_selection(train_val_df, 100, ('surv', 'mortality_90d'), args.result_path, args.seed)
        viz_utils.plot_feat_importance(args.result_path, 'mortality_90d')
        selected_obse_cols = [('obse', col[1]) for col in df.columns if col[1] in surv_feats]
        non_obse_cols =  [col for col in df.columns if col[0] != 'obse']
        combined_cols = non_obse_cols + selected_obse_cols
        df = df.reindex(columns=pd.MultiIndex.from_tuples(combined_cols))

    if args.model == 'best_k':
        exp_path = join(args.result_path, args.exp_name, args.dataset)
        best_k(exp_path, df[df[('meta', 'group')] == 2])
    elif args.model == 'viz_cluster':
        result_path = join(args.result_path, args.exp_name, args.dataset, 'cluster4', 'seed0')
        viz_cluster(result_path, df[df[('meta', 'group')] == 2])
    elif args.model == 'explain':
        result_path = join(args.result_path, args.exp_name, args.dataset, 'cluster4', 'seed0')
        explain(result_path, df)
    elif args.model == 'seqVaGMM':
        # train models
        save_path = join(args.result_path, args.exp_name, args.dataset, 'cluster' + str(args.n_clusters), 'seed' + str(args.seed))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        SequantialVariationalGMM(df, save_path)
    else:
        assert False, 'Unknown model!'