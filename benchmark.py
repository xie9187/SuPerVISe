import os
import time
import copy
import argparse
import platform
import numpy as np
import pandas as pd
import joblib
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
parser.add_argument('--n_clusters', type=int, default=5)
parser.add_argument('--learn_prior', type=bool, default=True)
parser.add_argument('--reduce_lr', type=bool, default=False)
parser.add_argument('--seq_att', type=bool, default=False)
parser.add_argument('--init_model', type=bool, default=False)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dim_red', type=str, default='tsne')
parser.add_argument('--exp_name', type=str, default='benchmark')
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
        if args.init_model:
            from tslearn.clustering import TimeSeriesKMeans
            print('training TSKM')
            clus_predictor = TimeSeriesKMeans(
                n_clusters=args.n_clusters,
                init='k-means++',
                n_init=1,
                verbose=0,
                random_state=args.seed,
                metric='euclidean',
            )
            model = seqVaGMM.init_model(
                df,
                model,
                lr=1e-3,
                decay=args.decay,
                b_size=64,
                n_epoch=1,
                clus_predictor=clus_predictor,
                seed=args.seed
            )
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
    

    thresh = None
    for i, set_name in enumerate(['valid', 'test']):
        print(set_name)
        sub_df = df[df[('meta', 'group')] == i + 1]
        X, y, seq_len = data_utils.create_numpy_data(sub_df, args.max_seq_len)
        dec, z_sample, p_c_z, risk, p_z_c = seqVaGMM.test_model(model, X, np.expand_dims(y, axis=-1), seq_len, 64)
        risk = np.squeeze(risk)
        thresh = evaluate(x=X, y_true=y, y_pred=risk, clus_pred=p_c_z, thresh=thresh, save_path=save_path, set_name=set_name)

def TPhenotypeClustering(df, save_path):
    from models.tphenotype import LaplaceEncoder, Predictor
    from models.tphenotype.utils.utils import create_range_mask, save_model, load_model

    Encoder_config = {
        'num_poles': 4,    # number of poles
        'max_degree': 2,    # maximum degree of poles
        'hidden_size': 20,    # number of hidden units in neural networks
        'num_layers': 1,    # number of layers in MLP of the encoder (1 layer RNN + n layer MLP)
        'pole_separation': 2.0,    # minimum distance between distinct poles 
        'freq_scaler': 20,    # scale up the imaginary part to help learning
        'window_size': None,    # whether or not to include time delay terms
        'equivariant_embed': True,    # whether or not to sort the poles (useless during training)
        'device': 'cpu',
    }

    Cls_config = {
        'K': None, 'steps': [-1], 'tol': 1e-6, 'test_num': 50, }

    Predictor_config = {
        'x_dim': None, 'y_dim': None, 'time_series_dims': None, 'hidden_size': 20, 'num_layer': 3,
        'global_bias': False, 'encoder_config': None, 'cls_config': None, 'categorical': True, 'device': 'cpu',
        'random_state': args.seed
    }

    loss_weights = {'ce': 1.0, 'rmse': 1.0, 'cont': 0.01, 'pole': 1.0, 'real': 0.1}

    datasets = []
    for i, set_name in enumerate(['train', 'valid', 'test']):
        sub_df = df[df[('meta', 'group')] == i]
        X, y, seq_len = data_utils.create_numpy_data(sub_df, args.max_seq_len)
        X, y = X.astype(np.float32), y.astype(np.float32)
        # Tphenotype needs the y shape to be (n_sample, seq_len, n_outcome)
        y_onehot = np.c_[1 - y, y].astype(np.float32) # (n_sample, n_outcome)
        y_seq_onehot = np.expand_dims(y_onehot, axis=1) # (n_sample, 1, n_outcome)
        y_seq_onehot = np.tile(y_seq_onehot, (1, args.max_seq_len, 1))

        # t is a 2d array (n_sample, seq_len) indicating the timestamp of each point
        t = np.tile(
                np.expand_dims(
                    np.arange(args.max_seq_len), axis=0
                ), (X.shape[0], 1)
            )

        mask = np.sum(np.fabs(X), axis=-1)
        mask[mask > 0] = 1
        
        range_mask = create_range_mask(mask)

        datasets.append({'x': X, 'y_raw': y, 'y': y_seq_onehot, 't':t, 'mask': mask, 'range_mask': range_mask})

    train_set, valid_set, test_set = datasets

    _, T, x_dim = train_set['x'].shape
    _, _, y_dim = train_set['y'].shape

    temporal_dims = [2, 3]

    encoder_config = Encoder_config.copy()
    encoder_config['pole_separation']=2.0
    encoder_config['max_degree'] = 2
    cls_config = Cls_config.copy()
    cls_config['K'] = args.n_clusters
    cls_config['steps'] = [-1]
    predictor_config = Predictor_config.copy()
    predictor_config['x_dim'] = x_dim
    predictor_config['y_dim'] = y_dim
    predictor_config['time_series_dims'] = temporal_dims
    predictor_config['cls_config'] = cls_config
    predictor_config['encoder_config'] = encoder_config

    model = Predictor(**predictor_config)

    model_path = os.path.join(save_path, 'model.pt')
    if os.path.exists(model_path):
        ckpt = load_model(model_path)
        model.load_state_dict(ckpt['model_state_dict'])
        print('load model from: ' + model_path)
        print(f'stage 3 - clustering on similarity graph')
        model.fit_clusters(train_set, verbose=True)
        
    else:
        model = model.fit(train_set, loss_weights, learning_rate=0.01,valid_set=valid_set, epochs=30, tolerance=None)
        save_model({'model_state_dict':model.state_dict()}, model_path)

    thresh = None
    for i, set_name in enumerate(['valid', 'test']):
        start_time = time.time()
        print(set_name)
        risk, clus_prob = model.test_model(datasets[i + 1])
        X = datasets[i + 1]['x']
        y = datasets[i + 1]['y_raw']
        thresh = evaluate(x=X, y_true=y, y_pred=risk, clus_pred=clus_prob, thresh=thresh, save_path=save_path, set_name=set_name)
        print('test time: %.1f min' % ((time.time() - start_time) / 60.))

def CAMELOTClustering(df, save_path):
    import models.camelot.model as CAMELOT 
    import tensorflow as tf

    data_dict = {}
    for i, set_name in enumerate(['train', 'valid', 'test']):
        sub_df = df[df[('meta', 'group')] == i]
        X, y, seq_len = data_utils.create_numpy_data(sub_df, args.max_seq_len)
        # CAMELOT needs one-hot encoded y label
        data_dict[set_name] = (X.astype(np.float32), np.c_[1 - y, y].astype(np.float32))

    data_info = {
        "X": (data_dict['train'][0], data_dict['valid'][0], data_dict['test'][0]),
        "y": (data_dict['train'][1], data_dict['valid'][1], data_dict['test'][1]),
        "data_load_config": {"data_name": "MIMIC"},
        "save_path": join(save_path, 'log')
    }

    model_config = { 
       "num_clusters": args.n_clusters, "latent_dim": 32, "seed": args.seed, "name": "ABL1", 
       "alpha_1": 0.0, "alpha_2": 0.1, "alpha_3": 0.0, "beta": 0.1,
       "regulariser_params": [0.01, 0.01], "dropout": 0.3,
       "encoder_params": {"hidden_layers": 1, "hidden_nodes": 30},
       "identifier_params": {"hidden_layers": 1, "hidden_nodes": 30},
       "predictor_params": {"hidden_layers": 2, "hidden_nodes": 30},
       "seed": args.seed,
    }
    training_config = {
       "lr_init": 0.002, "lr": 0.001, "epochs_init_1": 10, "epochs_init_2": 10,
       "epochs": 30, "bs": 64, "cbck_str": "", "patience_epochs": 200, "gpu": 0
    }

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    with tf.device('/cpu'):
        model = CAMELOT.Model(data_info=data_info, model_config=model_config, training_config=training_config)
        model.train(data_info=data_info, save_path=save_path, **training_config)

    thresh = None
    for i, set_name in enumerate(['valid', 'test']):
        print(set_name)
        sub_df = df[df[('meta', 'group')]==i + 1]
        X, y, seq_len = data_utils.create_numpy_data(sub_df, args.max_seq_len)
        risk, clus_prob = model.test_model(X, args.b_size)
        thresh = evaluate(x=X, y_true=y, y_pred=risk, clus_pred=clus_prob, thresh=thresh, save_path=save_path, set_name=set_name)

def ACTPCClustering(df, save_path):
    import models.actpc.model as ACTPC
    import tensorflow as tf

    data_dict = {}
    for i, set_name in enumerate(['train', 'valid', 'test']):
        sub_df = df[df[('meta', 'group')] == i]
        X, y, seq_len = data_utils.create_numpy_data(sub_df, args.max_seq_len)
        data_dict[set_name] = (X.astype(np.float32), np.c_[1 - y, y].astype(np.float32))

    data_info = {
        "X": (data_dict['train'][0], data_dict['valid'][0], data_dict['test'][0]),
        "y": (data_dict['train'][1], data_dict['valid'][1], data_dict['test'][1]),
        "data_load_config": {"data_name": "MIMIC"},
        "save_path": join(save_path, 'log')
    }

    model_config = { 
        "num_clusters": args.n_clusters, "latent_dim": 32, "seed": args.seed, "name": "ACTPC", 
        "alpha": 0.01, "beta": 0.01,
        "regulariser_params": [0.01, 0.01], "dropout": 0.6,
        "Actor_params": {"hidden_layers": 1, "hidden_nodes": 50, "state_fn": "tanh"},
        "Selector_params": {"hidden_layers": 2, "hidden_nodes": 100, "activation_fn": "relu"},
        "Critic_params": {"hidden_layers": 2, "hidden_nodes": 100, "activation_fn": "relu"},
        "cluster_rep_lr": 0.01
    }
    training_config = {
       "lr_init": 0.001, "lr": 0.001, "epochs_init": 10,
       "epochs": 10, "bs": 64, "cbck_str": "", "patience_epochs": 200, "gpu": 0
    }
    
    with tf.device('/cpu'):
        model = ACTPC.Model(data_info=data_info, model_config=model_config, training_config=training_config)
        model.train(data_info=data_info, save_path=save_path, **training_config)

    thresh = None
    for i, set_name in enumerate(['valid', 'test']):
        print(set_name)
        sub_df = df[df[('meta', 'group')]==i + 1]
        X, y, seq_len = data_utils.create_numpy_data(sub_df, args.max_seq_len)
        risk, clus_prob = model.test_model(X)
        thresh = evaluate(x=X, y_true=y, y_pred=risk, clus_pred=clus_prob, thresh=thresh, save_path=save_path, set_name=set_name)

def TSKMClustering(df, save_path):
    from tslearn.clustering import TimeSeriesKMeans

    model = TimeSeriesKMeans(
        n_clusters=args.n_clusters,
        init='k-means++',
        n_init=1,
        verbose=0,
        random_state=args.seed,
        metric='euclidean',
    )

    thresh = None
    for i, set_name in enumerate(['train', 'valid', 'test']):
        print(set_name)
        sub_df = df[df[('meta', 'group')] == i]
        X, y, seq_len = data_utils.create_numpy_data(sub_df, args.max_seq_len)

        if set_name == 'train':
            start_time = time.time()
            model.fit(X)
            print('time: {:.1f} sec.'.format(time.time() - start_time))
        else:
            clus_pred = model.predict(X)
            clus_probs = np.eye(args.n_clusters)[clus_pred]
            clus_data_phens = np.zeros((args.n_clusters))
            for clus_id in range(args.n_clusters):
                clus_data_phens[clus_id] = np.sum(y[clus_pred == clus_id], axis=0)
            risk = clus_data_phens[clus_pred]
            thresh = evaluate(x=X, y_true=y, y_pred=risk, clus_pred=clus_probs, thresh=thresh, save_path=save_path, set_name=set_name)

def TraditionalClassifier(df, save_path):
    if args.model == 'RandomForest':
        from sklearn.ensemble import RandomForestClassifier 

        model = RandomForestClassifier(
            n_estimators=1000,
            max_depth=None,
            n_jobs=-1,
            random_state=args.seed
        )
    elif args.model == 'XGBoost':
        import xgboost as xgb

        model = xgb.XGBClassifier(
            n_estimators=1000, 
            max_depth=None,
            objective='binary:logistic',
            n_jobs=-1,
            random_state=args.seed
        )

    thresh = None
    for i, set_name in enumerate(['train', 'valid', 'test']):
        print(set_name)
        sub_df = df[df[('meta', 'group')] == i]
        X, y, seq_len = data_utils.create_numpy_data(sub_df, args.max_seq_len)
        X_reshape = np.reshape(X, (X.shape[0], -1))
        if set_name == 'train':
            start_time = time.time()
            model.fit(X_reshape, y)
            print('time: {:.1f} sec.'.format(time.time() - start_time))
        else:
            risk = model.predict_proba(X_reshape)[:, 1]
            clus_pred = data_utils.stratify_patients(risk, args.n_clusters)
            clus_probs = np.eye(args.n_clusters)[clus_pred]
            thresh = evaluate(x=X, y_true=y, y_pred=risk, clus_pred=clus_probs, thresh=thresh, save_path=save_path, set_name=set_name)

def summary(exp_path):
    # model_folders = sorted([x for x in os.listdir(exp_path) if '.' not in x])
    model_folders = ['RandomForest', 'XGBoost', 'TSKM', 'ACTPC', 'CAMELOT', 'TPhenotype', 'seqVaGMM-100epoch']
    df_list = []
    for model_folder in model_folders:
        seed_path = join(exp_path, model_folder)
        seed_folders = sorted([x for x in os.listdir(seed_path) if '.' not in x])
        df = pd.concat([pd.read_csv(join(seed_path, folder, 'test_result.csv'), index_col=0) for folder in seed_folders])
        df_mean = df.mean(axis=0)
        df_std = df.std(axis=0) 
        # Combine mean and std into the desired format for each column
        val_str = df_mean.apply("{:.3f}".format) + '±' + df_std.apply("{:.3f}".format)
        result_df = pd.DataFrame([val_str], index=[model_folder])
        df_list.append(result_df)
    result_df = df = pd.concat(df_list)
    result_df.to_csv(join(exp_path, 'test_result.csv'))

if __name__ == '__main__':
    np.random.seed(args.seed)

    if args.model == 'summary':
        exp_path = join(args.result_path, args.exp_name, args.dataset)
        summary(exp_path)
        exit() 
    else:
        # load data
        if 'mimic-iii2iv' in args.dataset:
            df1, _ = data_utils.load_data(args.data_path, args.seed)
            df2, _ = data_utils.load_data(args.data_path2, args.seed)
            df1[df1[('meta', 'group')] == 2] = 1
            df2[('meta', 'group')] = 2
            df = pd.concat([df1, df2], axis=0)
        else:
            df, _ = data_utils.load_data(args.data_path, args.seed)

        save_path = join(args.result_path, args.exp_name, args.dataset, args.model, 'seed' + str(args.seed))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if 'seqVaGMM' in args.model:
            SequantialVariationalGMM(df, save_path)
        elif 'TPhenotype' in args.model:
            TPhenotypeClustering(df, save_path)
        elif 'CAMELOT' in args.model:
            CAMELOTClustering(df, save_path)
        elif 'ACTPC' in args.model:
            ACTPCClustering(df, save_path)
        elif 'TSKM' in args.model:
            TSKMClustering(df, save_path)
        elif args.model in ['RandomForest', 'XGBoost']:
            TraditionalClassifier(df, save_path)
        else:
            assert False, 'Unknown Model!'