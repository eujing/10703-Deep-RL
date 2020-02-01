import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
import tensorflow_probability as tfp
# from keras.layers import Dense, Flatten, Input, Concatenate, Lambda, Activation
# from keras.models import Model
# from keras.regularizers import l2
import keras.backend as K
import numpy as np
import scipy.stats as stats
from util import ZFilter
import math
import pdb

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400
HIDDEN3_UNITS = 400

LOSS_CONST = math.log(2 * math.pi)

class PENN:
    """
    (P)robabilistic (E)nsemble of (N)eural (N)etworks
    """

    def __init__(self, num_nets, state_dim, action_dim, learning_rate):
        """
        :param num_nets: number of networks in the ensemble
        :param state_dim: state dimension
        :param action_dim: action dimension
        :param learning_rate:
        """

        self.num_nets = num_nets
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Log variance bounds
        self.max_logvar = tf.Variable(-3 * np.ones([1, self.state_dim]), dtype=tf.float32)
        self.min_logvar = tf.Variable(-7 * np.ones([1, self.state_dim]), dtype=tf.float32)

        # Create and initialize your model
        self.models = [self.create_network() for _ in range(num_nets)]
        self.optimizers = [tf.keras.optimizers.Adam(learning_rate=learning_rate) for _ in range(num_nets)]

    @tf.function
    def get_output(self, output):
        """
        Argument:
          output: tf variable representing the output of the keras models, i.e., model.output
        Return:
          mean and log variance tf tensors
        Note that you will still have to call sess.run on these tensors in order to get the
        actual output.
        """
        mean = output[:, 0:self.state_dim]
        raw_v = output[:, self.state_dim:]
        logvar = self.max_logvar - tf.nn.softplus(self.max_logvar - raw_v)
        logvar = self.min_logvar + tf.nn.softplus(logvar - self.min_logvar)
        return mean, logvar

    def create_network(self):
        return tf.keras.Sequential([
            Dense(HIDDEN1_UNITS, activation='relu', kernel_regularizer=l2(0.0001),
                input_shape=(self.state_dim + self.action_dim, )),
            Dense(HIDDEN2_UNITS, activation='relu', kernel_regularizer=l2(0.0001)),
            Dense(HIDDEN3_UNITS, activation='relu', kernel_regularizer=l2(0.0001)),
            Dense(2 * self.state_dim, activation='linear', kernel_regularizer=l2(0.0001))
        ])

    @tf.function
    def loss(self, mean, logvar, targets):
        assert mean.shape[1:] == (self.state_dim, )
        assert logvar.shape[1:] == (self.state_dim, )
        assert targets.shape[1:] == (self.state_dim, )
        assert len(mean) == len(logvar) == len(targets)

        var = tf.exp(logvar)
        weighted_l2 = tf.reduce_sum(tf.square(targets - mean) / var, axis=1)
        log_dets = tf.reduce_sum(logvar, axis=1)
        const = self.state_dim * LOSS_CONST

        return tf.reduce_mean(const + log_dets + weighted_l2, axis=0) / 2


    @tf.function
    def rmse(self, targets, pred_means):
        assert targets.shape[1:] == (self.state_dim, )
        assert pred_means.shape[1:] == (self.state_dim, )

        rmses = tf.sqrt(tf.reduce_mean(tf.square(targets - pred_means), axis=1))

        return tf.reduce_mean(rmses)


    def train(self, inputs, targets, batch_size=128, epochs=5, verbose=False):
        """
        Arguments:
          inputs: state and action inputs.  Assumes that inputs are standardized.
          targets: resulting states
        """
        assert inputs.shape[1:] == (self.state_dim + self.action_dim, )
        assert targets.shape[1:] == (self.state_dim, )
        inputs = inputs.astype(np.float32)
        targets = targets.astype(np.float32)

        if verbose:
            print(f"Training on {len(inputs)} transitions...")

        losses = [[] for _ in range(self.num_nets)]
        rmses = [[] for _ in range(self.num_nets)]
        for i in range(epochs):
            for n in range(self.num_nets):
                epoch_net_loss_avg = tf.keras.metrics.Mean()
                epoch_net_rmse_avg = tf.keras.metrics.Mean()
                model = self.models[n]
                optimizer = self.optimizers[n]

                # Resample with replacement for each network
                resamp_idx = np.random.choice(len(inputs), size=len(inputs))
                train_ds = tf.data.Dataset.from_tensor_slices((inputs[resamp_idx], targets[resamp_idx]))

                for input_batch, target_batch in train_ds.batch(batch_size):
                    with tf.GradientTape() as tape:
                        output = model(input_batch)
                        mean, logvar = self.get_output(output)
                        # loss = self.loss(mean, logvar, target_batch)
                        # Seems like its cheating, but probably more numerically stable
                        loss_check = tfp.distributions.MultivariateNormalDiag(mean, scale_diag=tf.exp(0.5 * logvar)).log_prob(target_batch)
                        loss_check = tf.negative(tf.reduce_mean(loss_check))
                        loss = loss_check
                        rmse = self.rmse(target_batch, mean)

                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
                    epoch_net_loss_avg(loss)
                    epoch_net_rmse_avg(rmse)
                if verbose:
                    print(f"Epoch {i} Net {n}: Mean Loss = {epoch_net_loss_avg.result()}, Mean RMSE = {epoch_net_rmse_avg.result()}")
                losses[n].append(float(epoch_net_loss_avg.result().numpy()))
                rmses[n].append(float(epoch_net_rmse_avg.result().numpy()))

        return losses, rmses

    # TODO: Write any helper functions that you need
