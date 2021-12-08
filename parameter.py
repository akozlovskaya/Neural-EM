# general
filename = '.\\data\\flying_mnist_hard_3digits.h5'
max_epoch = 500
gradient_gamma = False        # whether to back-propagate a gradient through gamma
nr_steps = 20               # number of (RN)N-EM steps
theta_size = 250
pixel_prior = {
        'mu': 0.0,              # mean of pixel prior Gaussian
        'sigma': 0.25           # std of pixel prior Gaussian
}
# em
k = 3                       # number of components
e_sigma = 0.25              # sigma used in the e-step when pixel distributions are Gaussian (acts as a temperature)
pred_init = 0.0             # initial prediction used to compute the input

noise_prob = 0.2            # probability of annihilating the pixel

batch_size = 64
lr = 0.0005

image_shape = (24, 24, 1)

train_size = 50000
valid_size = 10000
test_size = 10000

acc_str = " acc over epoch:\nseq_ARI = %.4f\nlast_ARI = %.4f\nseq_conf = %.4f\nlast_conf = %.4f\n"