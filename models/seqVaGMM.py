import os, time
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow import keras
from tensorflow.keras import layers
from models.tf_utils import *

import data_utils

tfd = tfp.distributions

class LSTMEncoder(layers.Layer):
    def __init__(self, encoded_size, hid_size=100):
        super().__init__(name='encoder')
        self.mask = layers.Masking(mask_value=0.)
        self.lstm = layers.LSTM(hid_size, activation='tanh')
        self.mu = layers.Dense(encoded_size)
        self.sigma = layers.Dense(encoded_size)

    def call(self, inputs):
        x = self.mask(inputs)
        x = self.lstm(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        return mu, sigma

class Attention(layers.Layer):
    def __init__(self, n_hid):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(n_hid) # input x weights
        self.W2 = tf.keras.layers.Dense(n_hid) # hidden states h weights
        self.V = tf.keras.layers.Dense(1) # V

    def call(self, features, hidden, seq_len):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
          
        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(features) + self.W2(hidden_with_time_axis))
        ) ## w[x, h]
        mask = tf.sequence_mask(seq_len, maxlen=tf.shape(features)[1])
        mask = tf.cast(mask, tf.bool)
        mask = tf.expand_dims(mask, axis=-1)
        score = tf.where(mask, score, tf.fill(tf.shape(score), -1e30))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1) ## v tanh(w[x,h])
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features # [batzh_size, max_length, hidden_size]
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class AttLSTMEncoder(layers.Layer):
    def __init__(self, encoded_size, hid_size):
        super().__init__(name='att_encoder')
        self.mask = layers.Masking(mask_value=0.)
        self.lstm = layers.LSTM(hid_size, return_sequences=True, return_state=True, activation='tanh')
        self.attention = Attention(10)
        self.mu = layers.Dense(encoded_size)
        self.sigma = layers.Dense(encoded_size)

    def call(self, inputs, seq_len):
        x = self.mask(inputs)
        x, h, c = self.lstm(x)
        x, a = self.attention(x, h, seq_len)
        mu = self.mu(x)
        sigma = self.sigma(x)
        return mu, sigma, a

class LSTMDecoder(layers.Layer):
    def __init__(self, feat_size, max_seq_len, hid_size=100):
        super().__init__(name='decoder')
        self.repeat = layers.RepeatVector(max_seq_len)
        self.dense = layers.Dense(hid_size)
        self.lstm = layers.LSTM(feat_size, return_sequences=True, activation=None)

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.repeat(x)
        out = self.lstm(x)
        return out

class seqVaGMM(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(name="seqVaGMM")

        tf.random.set_seed(kwargs['seed'])

        self.inp_shape = kwargs['inp_shape']
        self.encoded_size = kwargs['latent_dim']
        self.hid_size = kwargs['hid_dim']
        self.max_seq_len = kwargs['max_seq_len']
        self.num_clusters = kwargs['num_clusters']
        self.s = kwargs['monte_carlo']
        self.sample_surv = kwargs['sample_surv']
        self.learn_prior = kwargs['learn_prior']
        self.survival = kwargs['survival']
        self.seq_att = kwargs['seq_att']
        self.c_mu = tf.Variable(tf.initializers.GlorotNormal()(shape=[self.num_clusters, self.encoded_size]), name='mu')
        self.log_c_sigma = tf.Variable(tf.initializers.GlorotNormal()([self.num_clusters, self.encoded_size]), name='sigma')
        # Cluster-specific survival model parameters, we just change the survival distribution to a discrete one
        self.c_beta = tf.Variable(tf.initializers.GlorotNormal()(shape=[self.num_clusters, self.encoded_size]), name='beta')
        # # Weibull distribution shape parameter
        # self.weibull_shape = kwargs['weibull_shape']

        if self.seq_att:
            self.encoder = AttLSTMEncoder(self.encoded_size, self.hid_size)
        else:
            self.encoder = LSTMEncoder(self.encoded_size, self.hid_size)
        self.decoder = LSTMDecoder(self.inp_shape, self.max_seq_len, self.hid_size)

        if self.learn_prior:
            self.prior_logits = tf.Variable(tf.ones([self.num_clusters]), name="prior")
        else:
            self.prior = tf.constant(tf.ones([self.num_clusters]) * (1 / self.num_clusters), name='prior')
        self.use_t = tf.Variable([1.0], trainable=False)

    def call(self, inputs, training=True):
        # inputs include clinical features 
        # x (b_size, seq_len, inp_dim)
        # death d (b_size, 1)  with 0=surv, 1=dead
        # seqence length l (b_size)

        x, d, l = inputs
        enc_input = tf.cast(x, tf.float64) # (b_size, seq_len, inp_dim)
        if self.seq_att:
            z_mu, log_z_sigma, att = self.encoder(enc_input, l) # (b_size, n_hid)
        else:
            z_mu, log_z_sigma = self.encoder(enc_input) # (b_size, n_hid)
            att = tf.zeros((32, 20), dtype=tf.float64)
        tf.debugging.check_numerics(z_mu, message="z_mu")

        z = tfd.MultivariateNormalDiag(loc=z_mu, scale_diag=tf.math.sqrt(tf.math.exp(log_z_sigma)))
        if training:
            z_sample = z.sample(self.s) # (sample_size, b_size, n_hid)
        else:
            z_sample = tf.expand_dims(z_mu, 0) # (1, b_size, n_hid)

        tf.debugging.check_numerics(self.c_mu, message="c_mu")
        tf.debugging.check_numerics(self.log_c_sigma, message="c_sigma")
        c_sigma = tf.math.exp(self.log_c_sigma) # (n_cluster, n_hid)

        # p(z|c)
        p_z_c = tf.stack([tf.math.log(
            tfd.MultivariateNormalDiag(loc=tf.cast(self.c_mu[i, :], tf.float64),
                                       scale_diag=tf.math.sqrt(tf.cast(c_sigma[i, :], tf.float64))).prob(
                tf.cast(z_sample, tf.float64)) + 1e-60) for i in range(self.num_clusters)], axis=-1) # (1, b_size, n_cluster)
        tf.debugging.check_numerics(p_z_c, message="p_z_c")
        # prior p(c)
        if self.learn_prior:
            prior_logits = tf.math.abs(self.prior_logits) # (n_cluster)
            norm = tf.math.reduce_sum(prior_logits, keepdims=True)
            prior = prior_logits / (norm + 1e-60) # (n_cluster)
        else:
            prior = self.prior  # (n_cluster)
        tf.debugging.check_numerics(prior, message="prior")

        if self.survival:
            # Compute Weibull distribution's scale parameter, given z and c
            # Change it to the binomial distribution with n=1
            tf.debugging.check_numerics(self.c_beta, message="c_beta")
            if self.sample_surv:
                # lambda_z_c = tf.stack([weibull_scale(x=z_sample, beta=self.c_beta[i, :])
                #                         for i in range(self.num_clusters)], axis=-1)
                lambda_z_c = pred_risk(z_sample, self.c_beta, self.num_clusters) # (1, b_size, n_cluster)
            else:
                # lambda_z_c = tf.stack([weibull_scale(x=tf.stack([z_mu for i in range(self.s)], axis=0),
                #                                      beta=self.c_beta[i, :]) for i in range(self.num_clusters)], axis=-1)
                lambda_z_c = pred_risk(tf.stack([z_mu for _ in range(self.s)], axis=0), self.c_beta, self.num_clusters)
            tf.debugging.check_numerics(lambda_z_c, message="lambda_z_c")

            # Evaluate p(t|z,c), assuming t|z,c ~ Weibull(lambda_z_c, self.weibull_shape)
            # p_t_z_c = tf.stack([weibull_log_pdf(t=t, d=d, lmbd=lambda_z_c[:, :, i], k=self.weibull_shape)
            #                     for i in range(self.num_clusters)], axis=-1)
            p_t_z_c = tf.math.log(1e-30 + tf.where(d == 1, lambda_z_c, 1 - lambda_z_c)) # (1, b_size, n_cluster)
            p_t_z_c = tf.clip_by_value(p_t_z_c, -1e+30, 1e+30)
            tf.debugging.check_numerics(p_t_z_c, message="p_t_z_c")

            p_c_z = tf.math.log(tf.cast(prior, tf.float64) + 1e-30) + tf.cast(p_z_c, tf.float64) + tf.cast(p_t_z_c, tf.float64)  # (1, b_size, n_cluster)
        else:
            p_c_z = tf.math.log(tf.cast(prior, tf.float64) + 1e-30) + tf.cast(p_z_c, tf.float64)  # (1, b_size, n_cluster)

        p_c_z = tf.nn.log_softmax(p_c_z, axis=-1)  # (b_size, n_cluster)
        p_c_z = tf.math.exp(p_c_z)
        tf.debugging.check_numerics(p_c_z, message="p_c_z")

        loss_clustering = - tf.reduce_sum(tf.multiply(tf.cast(p_c_z, tf.float64), tf.cast(p_z_c, tf.float64)), axis=-1)

        loss_prior = - tf.math.reduce_sum(tf.math.xlogy(tf.cast(p_c_z, tf.float64), 1e-60 +
                                                                tf.cast(prior, tf.float64)), axis=-1)

        loss_variational_1 = - 1 / 2 * tf.reduce_sum(log_z_sigma + 1, axis=-1)

        loss_variational_2 = tf.math.reduce_sum(tf.math.xlogy(tf.cast(p_c_z, tf.float64),
                                                                      1e-60 + tf.cast(p_c_z, tf.float64)), axis=-1)

        if self.survival:
            loss_survival = -tf.reduce_sum(tf.multiply(tf.cast(p_t_z_c, tf.float64), tf.cast(p_c_z, tf.float64)), axis=-1)
            tf.debugging.check_numerics(loss_survival, message="loss_survival")
            self.add_loss(tf.math.reduce_mean(loss_survival))
            self.add_metric(loss_survival, name='loss_survival', aggregation="mean")

        tf.debugging.check_numerics(loss_clustering, message="loss_clustering")
        tf.debugging.check_numerics(loss_prior, message="loss_prior")
        tf.debugging.check_numerics(loss_variational_1, message="loss_variational_1")
        tf.debugging.check_numerics(loss_variational_2, message="loss_variational_2")
        
        self.add_loss(tf.math.reduce_mean(loss_clustering))
        self.add_loss(tf.math.reduce_mean(loss_prior))
        self.add_loss(tf.math.reduce_mean(loss_variational_1))
        self.add_loss(tf.math.reduce_mean(loss_variational_2))
        
        self.add_metric(loss_clustering, name='loss_clustering', aggregation="mean")
        self.add_metric(loss_prior, name='loss_prior', aggregation="mean")
        self.add_metric(loss_variational_1, name='loss_variational_1', aggregation="mean")
        self.add_metric(loss_variational_2, name='loss_variational_2', aggregation="mean")
        
        # we always only sample one z
        dec = self.decoder(z_sample[0]) # (b_size, seq_len, input_size)

        # masked rmse loss
        mask = create_mask_from_seq_len(l, self.max_seq_len)
        loss_reconstruction = masked_rmse_loss(y_true=x, y_pred=dec, mask=mask)
        tf.debugging.check_numerics(loss_reconstruction, message="loss_reconstruction")
        self.add_loss(tf.math.reduce_mean(loss_reconstruction))
        self.add_metric(loss_reconstruction, name='loss_reconstruction', aggregation="mean")

        # Evaluate risk scores based on hard clustering assignments
        # Survival time may ba unobserved, so a special procedure is needed when time is not observed...
        p_z_c = p_z_c[0]    # take the first sample
        p_c_z = p_c_z[0]

        if self.survival:
            lambda_z_c = lambda_z_c[0]  # Take the first sample, since there is only 1 MC sample
            # Use Bayes rule to compute p(c|z) instead of p(c|z,t), since t is unknown
            p_c_z_nt = tf.math.log(tf.cast(prior, tf.float64) + 1e-60) + tf.cast(p_z_c, tf.float64)
            p_c_z_nt = tf.nn.log_softmax(p_c_z_nt, axis=-1)
            p_c_z_nt = tf.math.exp(p_c_z_nt)
            inds_nt = tf.dtypes.cast(tf.argmax(p_c_z_nt, axis=-1), tf.int32)
            risk_scores_nt = tensor_slice(target_tensor=tf.cast(lambda_z_c, tf.float64), index_tensor=inds_nt)

            inds = tf.dtypes.cast(tf.argmax(p_c_z, axis=-1), tf.int32)
            risk_scores_t = tf.cast(tensor_slice(target_tensor=lambda_z_c, index_tensor=inds), dtype=tf.float64)

            p_c_z = tf.cond(self.use_t[0] < 0.5, lambda: p_c_z_nt, lambda: p_c_z)
            risk_prob = tf.cond(self.use_t[0] < 0.5, lambda: risk_scores_nt, lambda: risk_scores_t)

        else:
            inds = tf.dtypes.cast(tf.argmax(p_c_z, axis=-1), tf.int32)
            risk_prob = tensor_slice(target_tensor=p_c_z, index_tensor=inds)
            lambda_z_c = risk_prob

        p_z_c = tf.cast(p_z_c, tf.float64)
        z_sample = z_sample[0]
        risk_prob = tf.expand_dims(risk_prob, -1)
        return dec, z_sample, p_c_z, risk_prob, p_z_c, att

    def vae(self, inputs, training):
        x, l = inputs
        x = tf.cast(x, tf.float64) # (b_size, seq_len, inp_dim)
        if self.seq_att:
            z_mu, log_z_sigma, _ = self.encoder(x, l) # (b_size, n_hid)
        else:
            z_mu, log_z_sigma = self.encoder(x) # (b_size, n_hid)

        z = tfd.MultivariateNormalDiag(loc=z_mu, scale_diag=tf.math.sqrt(tf.math.exp(log_z_sigma)))
        if training:
            z_sample = z.sample(self.s)[0] # (b_size, n_hid)
        else:
            z_sample = z_mu # (b_size, n_hid)

        dec = self.decoder(z_sample) # (b_size, seq_len, input_size)
        return z_mu, log_z_sigma, dec

def train_model(
        df,
        model, 
        lr,
        decay,
        b_size,
        num_epochs,
        reduce_lr,
        save_path
    ):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, decay=decay)
    cp_callback = [
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(save_path, 'logs'))
    ]
    if reduce_lr:
        cp_callback.append(
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss_survival', factor=0.1, patience=5, min_lr=1e-5)
        )
    model.compile(optimizer, metrics={"output_4": tf.keras.metrics.BinaryAccuracy()})

    data_gens = []
    for group in range(3):
        sub_df = df[df[('meta', 'group')] == group]
        gen = SeqDataGen(sub_df, batch_size=b_size, max_seq_len=model.max_seq_len, shuffle=True)
        data_gens.append(gen)

    # model training
    tf.keras.backend.set_value(model.use_t, np.array([1.0]))
    start_time = time.time()
    model.fit(data_gens[0], validation_data=data_gens[1], callbacks=cp_callback, epochs=num_epochs, verbose=2)
    print('training {:} epochs, time used {:.1f}'.format(num_epochs, (time.time() - start_time) / 60.))

    return model

def test_model(model, X, y, seq_len, b_size):
    model.sample_surv = False
    tf.keras.backend.set_value(model.use_t, np.array([0.0]))
    dec, z_sample, p_c_z, risk, p_z_c, att = model.predict((X, y, seq_len), batch_size=b_size)
    return dec, z_sample, p_c_z, risk, p_z_c

def init_model(
        df, 
        model,
        lr, 
        decay,
        b_size,
        n_epoch,
        clus_predictor,
        seed,
    ):
    # initailize clusters
    sub_df = df[df[('meta', 'group')] == 0]
    X_tr, y_tr, seq_len_tr = data_utils.create_numpy_data(sub_df, model.max_seq_len)

    sub_df = df[df[('meta', 'group')] == 1]
    X_va, y_va, seq_len_va = data_utils.create_numpy_data(sub_df, model.max_seq_len)

    clus_predictor.fit(X_tr)
    clus_tr = clus_predictor.predict(X_tr)
    clus_va = clus_predictor.predict(X_va)

    optimiser = keras.optimizers.Adam(learning_rate=lr, decay=decay)

    # initialize encoder and decoder
    print("Initialising VAE")

    train_dataset = tf.data.Dataset.from_tensor_slices((X_tr, seq_len_tr)).shuffle(1000, seed=seed).batch(b_size)
    valid_dataset = tf.data.Dataset.from_tensor_slices((X_va, seq_len_va)).shuffle(1000, seed=seed).batch(b_size)

    # manually build model once
    model.vae((X_tr[:10], seq_len_tr[:10]), training=True)

    var_list = [var for var in model.trainable_variables if "encoder" in var.name or "decoder" in var.name]

    for epoch in range(n_epoch):
        # training
        epoch_loss, step_ = 0, 0
        start_time = time.time()
        for step_, (x_batch, seq_len_batch) in enumerate(train_dataset):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(var_list)

                z_mean, z_var, recon = model.vae((x_batch, seq_len_batch), training=True)
                mask = tf.sequence_mask(seq_len_batch, maxlen=tf.shape(x_batch)[1])
                recon_loss = masked_rmse_loss(tf.cast(x_batch, tf.float32), recon, mask)
                kl_loss = kl_divergence_loss(z_mean, z_var)
                loss_batch = recon_loss + kl_loss
            
            grad = tape.gradient(loss_batch, var_list)
            optimiser.apply_gradients(zip(grad, var_list))

            epoch_loss += loss_batch

        epoch_loss = epoch_loss / step_

        # validation
        val_loss, val_step_ = 0, 0
        for val_step_, (x_batch, seq_len_batch) in enumerate(valid_dataset):
            z_mean, z_var, recon = model.vae((x_batch, seq_len_batch), training=False)
            mask = tf.sequence_mask(seq_len_batch, maxlen=tf.shape(x_batch)[1])
            recon_loss = masked_rmse_loss(tf.cast(x_batch, tf.float32), recon, mask)
            kl_loss = kl_divergence_loss(z_mean, z_var)
            loss_val_batch = recon_loss + kl_loss

            val_loss += loss_val_batch

        # Take mean over whole dataset
        val_loss = val_loss / val_step_
        print("End of epoch %d - \n Training loss: %.4f  Validation loss %.4f  Epoch time %.2f " % (
            epoch, epoch_loss, val_loss, time.time() - start_time))

    # initialize c_mu, c_sigma and prior
    print("Initialising centroids")
    # training set
    z_list = []
    for ids in range(0, len(X_tr), b_size):
        x_batch = X_tr[ids : ids + b_size]
        seq_len_batch = seq_len_tr[ids]
        z_mean, _, _ = model.vae((x_batch, seq_len_batch), training=False)
        z_list.append(z_mean)
    z_tr = tf.concat(z_list, axis=0).numpy()

    # validation set
    z_list = []
    for ids in range(0, len(X_va), b_size):
        x_batch = X_va[ids : ids + b_size]
        seq_len_batch = seq_len_va[ids]
        z_mean, _, _ = model.vae((x_batch, seq_len_batch), training=False)
        z_list.append(z_mean)
    z_va = tf.concat(z_list, axis=0).numpy()
    
    c_mu_list, c_sigma_list, prior_list = [], [], []
    for cluster in range(model.num_clusters):
        if np.sum(clus_tr == cluster) > 0:
            z_clus = z_tr[clus_tr == cluster]
            c_mu = np.mean(z_clus, axis=0)
            c_sigma = np.log(np.var(z_clus, axis=0) + 1e-30)
        else:
            c_mu = np.zeros((z_tr.shape[1]))
            c_sigma = np.zeros((z_tr.shape[1]))
        c_mu_list.append(c_mu)
        c_sigma_list.append(c_sigma)
        prior_list.append(np.sum(clus_tr == cluster))

    c_mu = np.stack(c_mu_list)
    c_sigma = np.stack(c_sigma_list)
    prior = np.stack(prior_list) / np.sum(prior_list) 

    tf.keras.backend.set_value(model.c_mu, c_mu)
    tf.keras.backend.set_value(model.log_c_sigma, c_sigma)
    if model.learn_prior:
        tf.keras.backend.set_value(model.prior_logits, prior)

    # initialize beta
    print('Initialising beta')
    train_dataset = tf.data.Dataset.from_tensor_slices((z_tr, y_tr, clus_tr)).shuffle(100, seed=seed).batch(b_size)
    valid_dataset = tf.data.Dataset.from_tensor_slices((z_va, y_va, clus_va)).shuffle(100, seed=seed).batch(b_size)

    var_list = [var for var in model.trainable_variables if "beta" in var.name]
    
    for epoch in range(n_epoch):
        # Training
        epoch_loss, step_ = 0, 0
        risk_probs = []
        start_time = time.time()
        for step_, (z_batch, y_batch, c_batch) in enumerate(train_dataset):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(var_list)
                risk_all_clus = pred_risk(z_batch, model.c_beta, model.num_clusters) # (b_size, n_cluster)
                inds = tf.cast(c_batch, tf.int32)
                risk_prob = tensor_slice(target_tensor=risk_all_clus, index_tensor=inds)
                loss_batch = tf.reduce_mean(tf.square(risk_prob - tf.cast(y_batch, tf.float32)))
                risk_probs.append(risk_prob)
  
            grad = tape.gradient(loss_batch, var_list)
            optimiser.apply_gradients(zip(grad, var_list))
            epoch_loss += loss_batch

        # Take mean over whole data
        epoch_loss = epoch_loss / step_
        risk_probs = tf.concat(risk_probs, axis=0)
        epoch_acc = tf.reduce_mean(tf.cast(tf.cast(risk_probs > 0.5, tf.float64) == y_tr, tf.float64))

        # validation
        val_loss, val_step_ = 0, 0
        risk_probs = []
        for val_step_, (z_batch, y_batch, c_batch) in enumerate(valid_dataset):
            # Get forward pass
            risk_all_clus = pred_risk(z_batch, model.c_beta, model.num_clusters) # (b_size, n_cluster)
            inds = tf.cast(c_batch, tf.int32)
            risk_prob = tensor_slice(target_tensor=risk_all_clus, index_tensor=inds)
            loss_val_batch = tf.reduce_mean(tf.square(risk_prob - tf.cast(y_batch, tf.float32)))
            val_loss += loss_val_batch
            risk_probs.append(risk_prob)

        val_loss = val_loss / val_step_
        risk_probs = tf.concat(risk_probs, axis=0)
        val_acc = tf.reduce_mean(tf.cast(tf.cast(risk_probs > 0.5, tf.float64) == y_va, tf.float64))

        print("End of epoch %d - \n Training loss: %.4f  Validation loss %.4f" % (epoch, epoch_loss, val_loss))
        print("Training acc: %.4f  Validation acc %.4f  Epoch time %.2f " % (epoch_acc, val_acc, time.time() - start_time))

    return model
