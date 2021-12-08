import tensorflow as tf
import tensorflow_probability as tfp
from parameter import pixel_prior, nr_steps

# LOSS
def compute_prior(pixel_prior = pixel_prior):
    return tf.constant(pixel_prior['mu'], shape=(1, 1, 1, 1, 1), name='prior')
    
def get_loss_step_weights(nr_steps = nr_steps):
    return [1.0] * nr_steps

def gaussian_squared_error_loss(mu, sigma, x):
    return (((mu - x)**2) / (2 * tf.clip_by_value(sigma ** 2, 1e-6, 1e6))) + tf.math.log(tf.clip_by_value(sigma, 1e-6, 1e6))

def kl_loss_gaussian(mu1, mu2, sigma1, sigma2):
    return tf.math.log(tf.clip_by_value(sigma2/sigma1, 1e-6, 1e6)) + (sigma1 ** 2 + (mu1 - mu2) ** 2) / (2 * sigma2 ** 2) - 0.5

class TotalLoss():
    
    def __init__(self, loss_inter_weight = 1.0):
        self.prior = compute_prior()
        self.loss_inter_weight = loss_inter_weight
    
    def __call__(self, mu, gamma, target):
        intra_loss = gaussian_squared_error_loss(mu, 1.0, target)
        inter_loss = kl_loss_gaussian(mu, self.prior, 1.0, 1.0)
        batch_size = tf.cast(tf.shape(target)[0], tf.float32)
        intra_loss = tf.reduce_sum(intra_loss * tf.stop_gradient(gamma)) / batch_size
        inter_loss = tf.reduce_sum(inter_loss * (1. - tf.stop_gradient(gamma))) / batch_size
        total_loss = intra_loss + self.loss_inter_weight * inter_loss
        return total_loss