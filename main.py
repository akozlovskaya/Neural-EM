import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
import h5py
import time
import numpy as np
import warnings

from nem import NEMCell, add_noise
from network import InnerModel
from loss import TotalLoss, get_loss_step_weights
from metric import AdjustedRandIndex
from data import InputPipeLine
from parameter import max_epoch, nr_steps, filename, batch_size, k, image_shape, lr, theta_size, acc_str

nr_iters = nr_steps + 1
train_inputs = InputPipeLine('training', sequence_length=nr_iters, filename = filename, batch_size=batch_size)
valid_inputs = InputPipeLine('validation', sequence_length=nr_iters, filename = filename, batch_size=batch_size)
loss_fn = TotalLoss()
optimizer = keras.optimizers.Adam(learning_rate=lr)

weights = get_loss_step_weights()

train_acc_metric = AdjustedRandIndex(weights)
val_acc_metric = AdjustedRandIndex(weights)

# set up inner cells and nem cells
inner_cell = InnerModel()
nem_cell = NEMCell(inner_cell, input_shape=image_shape)

for epoch in range(max_epoch):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()
    
    # TRAINING
    train_inputs.set_zero_batch_state()
    for step in range(train_inputs.get_n_batches()):
    
        batch_train = train_inputs.get_batch()
        x_batch_train = batch_train['features']
        y_batch_train = batch_train['groups']
        x_batch_train_corrupted = add_noise(x_batch_train)
        
        # get state initializer
        hidden_state = nem_cell.init_state(batch_size, k)

        # build static iterations
        total_losses = 0
        thetas, preds, gammas = [], [], []
        with tf.GradientTape() as tape:
            for t, loss_weight in enumerate(weights):
            
                # run nem cell
                inputs = (x_batch_train_corrupted[t], x_batch_train[t+1])
                hidden_state, output = nem_cell(inputs, hidden_state)
                theta, pred, gamma = output
                thetas.append(tf.reshape(theta, (1, batch_size*k, theta_size)))
                preds.append(pred)
                gammas.append(gamma)
                
                # compute losses
                loss_value = loss_fn(pred, gamma, x_batch_train[t+1])
                total_losses += loss_weight * loss_value
                
            total_loss = total_losses/tf.cast(nr_steps, tf.float32)
        grads = tape.gradient(total_loss, nem_cell.model.trainable_weights)
        optimizer.apply_gradients(zip(grads, nem_cell.model.trainable_weights))
        
        thetas = tf.stack(thetas)               # (T, 1, B*K, M)
        preds = tf.stack(preds)                 # (T, B, K, W, H, C)
        gammas = tf.stack(gammas)               # (T, B, K, W, H, C)
        
        # Update training metric.
        train_acc_metric.update_state(y_batch_train[:nr_steps], gammas)


        if step % 50 == 0:
            print("\nTraining loss (for one batch) at step %d: %.4f" % (step, float(total_loss)))
            print("Seen so far: %s samples" % ((step + 1) * batch_size))
    
    # Display metrics at the end of each epoch.
    seq_ARI, last_ARI, seq_conf, last_conf = train_acc_metric.result()
    print("Training" + acc_str % (float(seq_ARI),float(last_ARI),float(seq_conf),float(last_conf)))

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()



    # VALIDATION
    valid_inputs.set_zero_batch_state()
    for step in range(valid_inputs.get_n_batches()):
        batch_valid = valid_inputs.get_batch()
        x_batch_valid = batch_valid['features']
        y_batch_valid = batch_valid['groups']
        x_batch_valid_corrupted = add_noise(x_batch_valid)
        
        hidden_state = nem_cell.init_state(batch_size, k)
        gammas = []
        for t, loss_weight in enumerate(weights):
            inputs = (x_batch_valid_corrupted[t], x_batch_valid[t+1])
            hidden_state, output = nem_cell(inputs, hidden_state)
            _, _, gamma = output
            gammas.append(gamma)
            
        gammas = tf.stack(gammas)
        val_acc_metric.update_state(y_batch_valid[:nr_steps], gammas)
        
    seq_ARI, last_ARI, seq_conf, last_conf = val_acc_metric.result()
    print("Validation" + acc_str % (float(seq_ARI),float(last_ARI),float(seq_conf),float(last_conf)))
    val_acc_metric.reset_states()
    print("Time taken: %.2fs" % (time.time() - start_time))