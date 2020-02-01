import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Concatenate, InputLayer, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import pdb

HIDDEN1_UNITS = 350
HIDDEN2_UNITS = 350


def create_critic_network(state_size, action_size, learning_rate,
        hidden_units_1=HIDDEN1_UNITS, hidden_units_2=HIDDEN2_UNITS):
    """Creates a critic network.

    Args:
        state_size: (int) size of the input.
        action_size: (int) size of the action.
        learning_rate: (float) learning rate for the critic.
    Returns:
        model: an instance of tf.keras.Model.
        state_input: a tf.placeholder for the batched state.
        action_input: a tf.placeholder for the batched action.
    """
    init = lambda: tf.keras.initializers.RandomUniform(minval=-3e-3, maxval=3e-3)
    state_input = Input(shape=(state_size, ))
    action_input = Input(shape=(action_size, ))
    layer1 = Dense(hidden_units_1, activation="linear",
            kernel_initializer=init(),
            bias_initializer=init())
    layer2_net = Dense(hidden_units_2, activation="linear",
            kernel_initializer=init(),
            bias_initializer=init())
    layer2_action = Dense(hidden_units_2, activation="linear",
            kernel_initializer=init(),
            bias_initializer=init())
    layer3 = Dense(1, activation="linear",
            kernel_initializer=init(),
            bias_initializer=init())

    net = tf.nn.relu(layer1(state_input))
    net = tf.nn.relu(layer2_net(net) + layer2_action(action_input))
    net = layer3(net)

    model = tf.keras.Model(inputs=[state_input, action_input], outputs=net)

    return model


class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, batch_size,
                 tau, learning_rate, hidden1, hidden2):
        """Initialize the CriticNetwork.
        This class internally stores both the critic and the target critic
        nets. It also handles computation of the gradients and target updates.

        Args:
            sess: A Tensorflow session to use.
            state_size: (int) size of the input.
            action_size: (int) size of the action.
            batch_size: (int) the number of elements in each batch.
            tau: (float) the target net update rate.
            learning_rate: (float) learning rate for the critic.
        """
        self.tau = tau
        self.optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        self.model = create_critic_network(state_size, action_size, learning_rate, hidden_units_1=hidden1, hidden_units_2=hidden2)
        self.target = create_critic_network(state_size, action_size, learning_rate, hidden_units_1=hidden1, hidden_units_2=hidden2)

        var_pairs = zip(
                self.model.trainable_variables,
                self.target.trainable_variables)

        for critic_param, target_param in var_pairs:
            target_param.assign(critic_param)


    def train(self, states, actions, targets, dbg=False):
        targets = targets.reshape(-1, 1)
        with tf.GradientTape() as tape:
            Q = self.model([states, actions], training=True)
            losses = tf.keras.losses.MSE(targets, Q)
            TD_losses = tf.abs(targets - Q)
        grads = tape.gradient(losses, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return losses, TD_losses

    def update_target(self):
        """Updates the target net using an update rate of tau."""
        var_pairs = zip(
                self.model.trainable_variables,
                self.target.trainable_variables)

        for critic_param, target_param in var_pairs:
            soft_update = self.tau * critic_param + (1 - self.tau) * target_param
            target_param.assign(soft_update)
