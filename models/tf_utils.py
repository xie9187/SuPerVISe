import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import roc_auc_score

tfd = tfp.distributions
tfkl = tf.keras.layers
tfpl = tfp.layers
tfk = tf.keras

class SeqDataGen(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=32, max_seq_len=20, shuffle=True):
        self.df = df
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle
        self.indexes = self.df[('meta', 'traj')].unique()
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Initialize lists to store batch data
        X = []
        y = []
        seq_len = []
        # Generate data
        for i in batch_indexes:
            # Filter data for one sequence
            sequence_data = self.df[self.df[('meta', 'traj')] == i]
            
            # Extract observations for the sequence
            X_seq = sequence_data.loc[:, ('obse', slice(None))].values
            
            # If sequence is longer than max_seq_len, truncate it
            if len(X_seq) > self.max_seq_len:
                X_seq = X_seq[:self.max_seq_len]

            # pad X to max_seq_len with zeros
            X_padded =  np.pad(X_seq, ((0, self.max_seq_len - len(X_seq)), (0, 0)), 'constant')
            X.append(X_padded)
            
            # Extract label for the sequence
            y_seq = sequence_data[('surv', 'mortality_90d')].iloc[-1]
            y.append(y_seq)

            seq_len.append(len(X_seq))
        
        X = np.array(X)
        y = np.expand_dims(np.array(y), axis=-1)
        seq_len = np.array(seq_len)
        return (X, y, seq_len), {"output_4": y}

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

def masked_rmse_loss(y_true, y_pred, mask):
    # y_true, y_pred (b_size, seq_len, input_size)
    # mask (b_size, seq_len)
    feat_loss = tf.square(y_true - y_pred) # (b_size, seq_len, input_size)
    sample_loss = tf.reduce_sum(feat_loss, axis=2) # (b_size, seq_len)
    seq_loss = tf.reduce_sum(
        sample_loss * tf.cast(mask, dtype=sample_loss.dtype), axis=1) / tf.reduce_sum(
        tf.cast(mask, dtype=sample_loss.dtype))  # (b_size)
    loss = tf.reduce_mean(sample_loss)
    return loss

def kl_divergence_loss(mean, log_var):
    kl_loss = -0.5 * (1 + log_var - tf.square(mean) - tf.exp(log_var))
    kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
    return kl_loss

def create_mask_from_seq_len(seq_len, max_seq_len):
    """
    Create a masking tensor for sequences given their lengths.

    Args:
        seq_len (tf.Tensor): A 1D tensor of size (batch_size,) containing the
                             sequence lengths of each batch item.
        max_seq_len (int): The maximum sequence length in the batch.

    Returns:
        tf.Tensor: A 2D boolean tensor of size (batch_size, max_seq_len) where
                   True indicates a valid sequence element and False indicates padding.
    """
    # Create a range tensor that matches the shape of the sequences
    range_tensor = tf.range(max_seq_len)  # [0, 1, 2, ..., max_seq_len-1]
    
    # Expand dims of seq_len and range_tensor to enable broadcasting
    seq_len_expanded = tf.expand_dims(seq_len, -1)  # (batch_size, 1)
    range_tensor_expanded = tf.expand_dims(range_tensor, 0)  # (1, max_seq_len)
    
    # Create mask by comparing range_tensor to seq_len
    mask = range_tensor_expanded < seq_len_expanded  # (batch_size, max_seq_len)
    
    return mask

def auc_metric(y_label, risk_scores):
    return tf.numpy_function(roc_auc_score, [y_label, risk_scores], tf.float64)

def tensor_slice(target_tensor, index_tensor):
    indices = tf.stack([tf.range(tf.shape(index_tensor)[0]), index_tensor], 1)
    return tf.gather_nd(target_tensor, indices)

def pred_risk(x, beta, n_c):
	return tf.sigmoid(
		tf.concat([
			tf.matmul(
				x, 
				tf.expand_dims(beta[i, :], axis=-1)
			) for i in range(n_c)], axis=-1
		)
	)

def pairwise_euclidean_distance(data_points, centroids):
    # Expand dimensions to enable broadcasting
    data_points_expanded = tf.expand_dims(data_points, axis=1)  # Shape: (batch_size, 1, hidden_dim)
    centroids_expanded = tf.expand_dims(centroids, axis=0)  # Shape: (1, cluster_num, hidden_dim)

    # Calculate squared Euclidean distance
    squared_distance = tf.reduce_sum(tf.square(data_points_expanded - centroids_expanded), axis=-1)  # Shape: (batch_size, cluster_num)

    # Take square root to get Euclidean distance
    distance = tf.sqrt(squared_distance)
    return distance
