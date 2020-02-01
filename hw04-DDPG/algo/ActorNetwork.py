import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, InputLayer, BatchNormalization, Dropout
from tensorflow.keras import Model, Sequential
from tensorflow.keras.regularizers import l2
import pdb

HIDDEN1_UNITS = 350
HIDDEN2_UNITS = 350


def create_actor_network(state_size, action_size,
        hidden_units_1=HIDDEN1_UNITS, hidden_units_2=HIDDEN2_UNITS):
    """Creates an actor network.

    Args:
        state_size: (int) size of the input.
        action_size: (int) size of the action.
    Returns:
        model: an instance of tf.keras.Model.
        state_input: a tf.placeholder for the batched state.
    """
    init = lambda: tf.keras.initializers.RandomUniform(minval=-3e-3, maxval=3e-3)
    return Sequential([
        InputLayer(input_shape=(state_size, )),
        Dense(hidden_units_1, activation="relu",
            kernel_initializer=init(),
            bias_initializer=init()),
        Dense(hidden_units_2, activation="relu",
            kernel_initializer=init(),
            bias_initializer=init()),
        Dense(action_size, activation="tanh",
            kernel_initializer=init(),
            bias_initializer=init())
    ])


class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, batch_size,
                 tau, learning_rate, hidden1, hidden2):
        """Initialize the ActorNetwork.
        This class internally stores both the actor and the target actor nets.
        It also handles training the actor and updating the target net.

        Args:
            sess: A Tensorflow session to use.
            state_size: (int) size of the input.
            action_size: (int) size of the action.
            batch_size: (int) the number of elements in each batch.
            tau: (float) the target net update rate.
            learning_rate: (float) learning rate for the critic.
        """
        self.model = create_actor_network(state_size, action_size, hidden_units_1=hidden1, hidden_units_2=hidden2)
        self.target = create_actor_network(state_size, action_size, hidden_units_1=hidden1, hidden_units_2=hidden2)
        self.batch_size = batch_size
        self.tau = tau
        self.optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

        var_pairs = zip(
                self.model.trainable_variables,
                self.target.trainable_variables)

        for actor_param, target_param in var_pairs:
            target_param.assign(actor_param)

    def train(self, states, critic, dbg=False):
        """Updates the actor by directly maximizing Q(s, a = actor(s | params))

        Args:
            states: a batched numpy array storing the state.
            critic: critic network
        """
        with tf.GradientTape() as tape:
            actions = self.model(states, training=True)
            Q = critic([states, actions], training=True)
            nQ = -1 * tf.reduce_mean(Q)

        if dbg:
            pdb.set_trace()

        # Optimizer minimizes, but we want to maximize, so negate
        grads = tape.gradient(nQ, self.model.trainable_variables)
        self.optimizer.apply_gradients(
                zip(grads, self.model.trainable_variables))
        return nQ

    def update_target(self):
        """Updates the target net using an update rate of tau."""
        var_pairs = zip(
                self.model.trainable_variables,
                self.target.trainable_variables)

        for actor_param, target_param in var_pairs:
            soft_update = self.tau * actor_param + (1 - self.tau) * target_param
            target_param.assign(soft_update)
