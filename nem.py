import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
import h5py
import time
import numpy as np
from parameter import pred_init, e_sigma, gradient_gamma, noise_prob

# NOISE
def add_noise(data, noise_prob = noise_prob):
    shape = tf.stack([s if s is not None else tf.shape(data)[i] for i, s in enumerate(data.get_shape())])
    noise_dist = tfp.distributions.Uniform(low=0.0, high=1.0)
    noise_uniform = noise_dist.sample(shape)

    # sample mask
    mask_dist = tfp.distributions.Bernoulli(probs=noise_prob, dtype=data.dtype)
    mask = mask_dist.sample(shape)

    # produce output
    corrupted = mask * noise_uniform + (1 - mask) * data

    corrupted.set_shape(data.get_shape())
    return corrupted


# EM
class NEMCell:
    def __init__(self, cell, input_shape, pred_init = pred_init, e_sigma = e_sigma):
        self.model = cell
        if not isinstance(input_shape, tf.TensorShape):
            input_shape = tf.TensorShape(input_shape)
        self.input_shape = input_shape
        self.gamma_shape = tf.TensorShape(input_shape.as_list()[:-1] + [1])
        self.pred_init = pred_init
        self.e_sigma = e_sigma

    @property
    def state_size(self):
        return self.model.state_size, self.input_shape, self.gamma_shape

    @property
    def output_size(self):
        return self.model.output_size, self.input_shape, self.gamma_shape

    def init_state(self, batch_size, K, dtype=tf.float32):
        # inner RNN hidden state init
        h = None

        # initial prediction (B, K, W, H, C)
        pred_shape = tf.stack([batch_size, K] + self.input_shape.as_list())
        pred = tf.ones(shape=pred_shape, dtype=dtype) * self.pred_init

        # initial gamma (B, K, W, H, 1)
        gamma_shape = self.gamma_shape.as_list()
        shape = tf.stack([batch_size, K] + gamma_shape)

        # init with Gaussian distribution
        gamma = tf.abs(tf.random.normal(shape, dtype=dtype))
        gamma /= tf.reduce_sum(gamma, 1, keepdims=True)

        return h, pred, gamma

    @staticmethod
    def delta_predictions(predictions, data):
        return data - predictions

    @staticmethod
    def mask_rnn_inputs(rnn_inputs, gamma, gradient_gamma = gradient_gamma):
        if not gradient_gamma:
            gamma = tf.stop_gradient(gamma)

        return rnn_inputs * gamma

    def run_inner_rnn(self, masked_deltas, h_old):
        shape = tf.shape(masked_deltas)
        batch_size = shape[0]
        K = shape[1]
        M = np.prod(self.input_shape.as_list())
        reshaped_masked_deltas = tf.reshape(masked_deltas, tf.stack([batch_size * K, M]))
        preds, h_new = self.model(reshaped_masked_deltas, h_old)
        return tf.reshape(preds, shape=shape), h_new

    def compute_em_probabilities(self, predictions, data, epsilon=1e-6):
        mu, sigma = predictions, self.e_sigma
        probs = ((1 / tf.sqrt((2 * np.pi * sigma ** 2))) * tf.exp(-(data - mu) ** 2 / (2 * sigma ** 2)))
        probs = tf.reduce_sum(probs, 4, keepdims=True, name='reduce_channels') + epsilon
        return probs

    def e_step(self, preds, targets):
        probs = self.compute_em_probabilities(preds, targets)
        # compute the new gamma (E-step)
        gamma = probs / tf.reduce_sum(probs, 1, keepdims=True)
        return gamma

    def __call__(self, inputs, state, scope=None):
        # unpack
        # input_data ~ (B, K, W, H, C)
        input_data, target_data = inputs
        h_old, preds_old, gamma_old = state

        # compute difference between prediction and input
        deltas = self.delta_predictions(preds_old, input_data)

        # mask with gamma
        masked_deltas = self.mask_rnn_inputs(deltas, gamma_old)

        # compute new predictions
        preds, h_new = self.run_inner_rnn(masked_deltas, h_old)

        # compute the new gammas
        gamma = self.e_step(preds, target_data)

        # pack and return
        outputs = (h_new, preds, gamma)
        return outputs, outputs