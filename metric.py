import tensorflow as tf
import numpy as np

"""
Inputs:
    groups: shape=(T, B, 1, W, H, 1)
    gammas: shape=(T, B, K, W, H, 1)
"""
    
class AdjustedRandIndex():

  def __init__(self, weights):
    self.seq_ARI = []
    self.last_ARI = []
    self.seq_conf = []
    self.last_conf = []
    self.w = weights

  def update_state(self, groups, gammas):
    groups = groups[1:]
    gammas = gammas[1:]
    # reshape gammas and convert to one-hot
    yshape = tf.shape(gammas)
    gammas = tf.reshape(gammas, shape=tf.stack([yshape[0] * yshape[1], yshape[2], yshape[3] * yshape[4] * yshape[5]]))
    Y = tf.one_hot(tf.argmax(gammas, axis=1), depth=yshape[2], axis=1)
    # reshape masks
    gshape = tf.shape(groups)
    groups = tf.reshape(groups, shape=tf.stack([gshape[0] * gshape[1], 1, gshape[3] * gshape[4] * gshape[5]]))
    G = tf.one_hot(tf.cast(groups[:, 0], tf.int32), depth=tf.cast(tf.reduce_max(groups) + 1, tf.int32), axis=1)
    # now Y and G both have dim (B*T, K, N) where N=W*H*C
    # mask entries with group 0
    M = tf.cast(tf.greater(groups, 0.5), tf.float32)
    n = tf.cast(tf.reduce_sum(M, axis=[1, 2]), tf.float32)
    #print('Y = ', Y)
    DM = G * M
    YM = Y * M
    # contingency table for overlap between G and Y
    nij = tf.einsum('bij,bkj->bki', YM, DM)
    a = tf.reduce_sum(nij, axis=1)
    b = tf.reduce_sum(nij, axis=2)
    # rand index
    rindex = tf.cast(tf.reduce_sum(nij * (nij-1), axis=[1, 2]), tf.float32)
    aindex = tf.cast(tf.reduce_sum(a * (a-1), axis=1), tf.float32)
    bindex = tf.cast(tf.reduce_sum(b * (b-1), axis=1), tf.float32)
    expected_rindex = aindex * bindex / (n*(n-1) + 1e-6)
    max_rindex = (aindex + bindex) / 2
    ARI = (rindex - expected_rindex)/tf.clip_by_value(max_rindex - expected_rindex, 1e-6, 1e6)
    ARI = tf.reshape(ARI, shape=(yshape[0], yshape[1]))
    iter_weigths= tf.constant(np.array(self.w)[:, None], dtype=tf.float32)
    sum_weights = tf.constant(np.sum(self.w), dtype=tf.float32)
    seq_ARI = tf.reduce_mean(tf.reduce_sum(ARI, axis=0) / sum_weights)
    last_ARI = tf.reduce_mean(ARI[-1])
    confidences = tf.reduce_sum(tf.reduce_max(gammas, axis=1, keepdims=True) * M, axis=[1, 2]) / n
    confidences = tf.reshape(confidences, shape=(yshape[0], yshape[1]))
    seq_conf = tf.reduce_mean(tf.reduce_sum(confidences, axis=0) / sum_weights)
    last_conf = tf.reduce_mean(confidences[-1])
    self.seq_ARI.append(seq_ARI)
    self.last_ARI.append(last_ARI)
    self.seq_conf.append(seq_conf)
    self.last_conf.append(last_conf)

  def result(self):
    return (tf.reduce_mean(tf.stack(self.seq_ARI)),
            tf.reduce_mean(tf.stack(self.last_ARI)),
            tf.reduce_mean(tf.stack(self.seq_conf)),
            tf.reduce_mean(tf.stack(self.last_conf)))

  def reset_states(self):
    self.seq_ARI = []
    self.last_ARI = []
    self.seq_conf = []
    self.last_conf = []
