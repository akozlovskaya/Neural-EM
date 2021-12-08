import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from parameter import batch_size, k, theta_size

# CUSTOM LAYER
class Conv2dResized(layers.Layer):
    def __init__(self, units=32, kernel_size = (4, 4), strides = [2, 2], activation = None):
        super(Conv2dResized, self).__init__()
        self.units = units
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        
        self.conv = layers.Conv2D(self.units, kernel_size = self.kernel_size, activation = self.activation, padding = 'same')

    def __call__(self, inputs):
        shape = inputs.get_shape()
        resized = tf.image.resize(inputs, (self.strides[0] * shape[1], self.strides[1] * shape[2]))
        output = self.conv(resized)
        return output

# NETWORK
class InnerModel(tf.keras.Model):

    def __init__(self, B = batch_size, K = k, theta_size = theta_size):
        super(InnerModel, self).__init__()
        self.enc_cell_0 = layers.LayerNormalization()
        self.enc_cell_1 = layers.Reshape(target_shape = (-1, 24, 24, 1))
        self.enc_cell_2 = layers.Conv2D(32, kernel_size = (4, 4), strides = [2, 2], activation = 'elu')
        self.enc_cell_3 = layers.Conv2D(64, kernel_size = (4, 4), strides = [2, 2], activation = 'elu')
        self.enc_cell_4 = layers.Conv2D(128, kernel_size = (4, 4), strides = [2, 2], activation = 'elu')
        self.enc_cell_5 = layers.Flatten()
        self.enc_cell_6 = layers.Dense(512, activation = 'elu')
        self.enc_cell_7 = layers.Reshape(target_shape = (512, 1))

        self.my_rnn = layers.SimpleRNN(250, activation='sigmoid', return_state=True)
        
        self.dec_cell_1 = layers.Dense(512, activation = 'relu')
        self.dec_cell_2 = layers.Dense(3*3*128, activation = 'relu')
        self.dec_cell_3 = layers.Reshape(target_shape = (3, 3, 128))
        self.dec_cell_4 = Conv2dResized(64, activation = 'relu')
        self.dec_cell_5 = Conv2dResized(32, activation = 'relu')
        self.dec_cell_6 = Conv2dResized(1)
        self.dec_output = layers.Flatten()
        
        
        
    def call(self, inputs, thetas_old=None, training=False):
        x = self.enc_cell_0(inputs)
        x = self.enc_cell_1(x)
        x = self.enc_cell_2(x)
        x = self.enc_cell_3(x)
        x = self.enc_cell_4(x)
        x = self.enc_cell_5(x)
        x = self.enc_cell_6(x)
        x = self.enc_cell_7(x)

        x, thetas_old = self.my_rnn(x, initial_state = thetas_old)

        x = self.dec_cell_1(x)
        x = self.dec_cell_2(x)
        x = self.dec_cell_3(x)
        x = self.dec_cell_4(x)
        x = self.dec_cell_5(x)
        x = self.dec_cell_6(x)
        x = self.dec_output(x)
    
        return x, thetas_old