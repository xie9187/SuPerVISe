import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
import argparse
from os.path import join

import models.seqVaGMM as seqVaGMM

parser = argparse.ArgumentParser(description='Process some parameters.')
parser.add_argument('--max_seq_len', type=int, default=20)
parser.add_argument('--n_emb', type=int, default=32)
parser.add_argument('--n_hid', type=int, default=100)
parser.add_argument('--n_clusters', type=int, default=4)
parser.add_argument('--learn_prior', type=bool, default=False)
parser.add_argument('--seq_att', type=bool, default=False)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--result_path', default='A path to the SuPerVISe')

args = parser.parse_args()

with open(join(args.result_path, 'data', 'test_data.p'), 'rb') as f:
    X, y, seq_len = pickle.load(f)

# X: 100 exmaple trajectories with 43 normalised clinical variables
# y: binary 90-day mortality
# seq_len: the length of the trajectories

with tf.device('/gpu'):
    model = seqVaGMM.seqVaGMM(
        latent_dim=args.n_emb,
        hid_dim=args.n_hid,
        num_clusters=args.n_clusters,
        inp_shape=X.shape[2],
        max_seq_len=args.max_seq_len,
        survival=True,
        monte_carlo=1,
        sample_surv=False,
        learn_prior=args.learn_prior,
        seq_att=args.seq_att,
        seed=args.seed,
    )
model_path = join(args.result_path, 'ckpt', 'model.ckpt')
model.load_weights(model_path).expect_partial()
print('load model from: ' + model_path)

# although we put y into the model here, since we set the variable use_t = 0 in the test_model function, the y will not be used.
dec, z, p_s_z, risk, p_z_s = seqVaGMM.test_model(model, X, np.expand_dims(y, axis=-1), seq_len, 64)
clus_pred = np.argmax(p_s_z, axis=-1)
print(clus_pred)