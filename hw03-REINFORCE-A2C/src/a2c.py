import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import keras.layers as layers
import keras.models as models
import gym
import json
import pdb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reinforce import Reinforce

def make_actor_model(input_size, output_size):
    init = lambda: keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform")
    model = models.Sequential([
        layers.Dense(16, kernel_initializer=init(), input_shape=input_size),
        layers.Activation("relu"),
        layers.Dense(16, kernel_initializer=init()),
        layers.Activation("relu"),
        layers.Dense(16, kernel_initializer=init()),
        layers.Activation("relu"),
        layers.Dense(output_size, activation="softmax", kernel_initializer=init()),
        ])

    return model

def make_critic_model(input_size):
    init = lambda: keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform")
    model = models.Sequential([
        layers.Dense(16, kernel_initializer=init(), input_shape=input_size),
        layers.Activation("relu"),
        layers.Dense(16, kernel_initializer=init()),
        layers.Activation("relu"),
        layers.Dense(16, kernel_initializer=init()),
        layers.Activation("relu"),
        layers.Dense(1, activation="linear", kernel_initializer=init()),
        ])

    return model

def shift(x, n):
    # Shift elements in a vector x to the left by n positions, filling in with zeros
    T = len(x)
    y = np.zeros_like(x)
    if n <= T:
        y[:(T - n)] = x[n:]
    return y

class A2C(Reinforce):
    # Implementation of N-step Advantage Actor Critic.
    # This class inherits the Reinforce class, so for example, you can reuse
    # generate_episode() here.

    def __init__(self, model, lr, critic_model, critic_lr, n=20):
        # Initializes A2C.
        # Args:
        # - model: The actor model.
        # - lr: Learning rate for the actor model.
        # - critic_model: The critic model.
        # - critic_lr: Learning rate for the critic model.
        # - n: The value of N in N-step A2C.
        self.model = model
        self.critic_model = critic_model
        self.n = n

        # Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.
        actor_opt = keras.optimizers.Adam(lr=lr)
        critic_opt = keras.optimizers.Adam(lr=critic_lr)
        self.model.compile(optimizer=actor_opt, loss="categorical_crossentropy")
        self.critic_model.compile(optimizer=critic_opt, loss="mse")

    def train(self, env, gamma=1.0, min_batch_size=200):
        # Trains the model on a single episode using A2C.

        all_states, all_actions, all_rewards = [], [], []
        all_V, all_R = [], []
        N = self.n
        gammaN = gamma**N

        # Potentially collect across more than 1 episode so we can maintain a minimum batch size
        while len(all_states) < min_batch_size:
            # Collect exactly 1 episode
            states, actions, rewards = self.generate_episode(env)
            all_states.extend(states)
            all_actions.extend(actions)
            all_rewards.extend(rewards)

            # Calculate R by episode, not across a few episodes
            T = len(rewards)
            G = np.zeros(T)
            R = np.zeros(T)

            # Predicted value of each state
            values = self.critic_model.predict(np.array(states)).squeeze()
            all_V.extend(values)
            values_end = shift(values, N)

            # Calculate G first
            G[-1] = rewards[-1]
            for i in reversed(range(T - 1)):
                G[i] = rewards[i] + gamma * G[i + 1]

            # Downscale rewards so network can learn more easily
            G /= 100

            # Calculate R from G, as R_t = G_t - gammaN * G_{t+N}
            G_excess = shift(G, N)

            # Actual formula for R
            # R = G - gammaN * G_excess + gammaN * values_end

            # Equivalent formula with less operations
            R = G + gammaN * (values_end - G_excess)

            all_R.extend(R)

        states, actions, rewards = np.array(all_states), np.array(all_actions), np.array(all_rewards)
        values, R = np.array(all_V), np.array(all_R)

        # OHE actions, but use sample_weight in fit() to reinforce with advantage
        targets = np.zeros((len(actions), env.action_space.n))
        targets[range(len(actions)), actions] = 1

        actor_history = self.model.fit(
                states, targets, sample_weight=(R - values),
                batch_size=1024, epochs=1, verbose=0)
        critic_history = self.critic_model.fit(
                states, R,
                batch_size=1024, epochs=1, verbose=0)
        return actor_history, critic_history


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The actor's learning rate.")
    parser.add_argument('--critic-lr', dest='critic_lr', type=float,
                        default=1e-4, help="The critic's learning rate.")
    parser.add_argument('--n', dest='n', type=int,
                        default=20, help="The value of N in N-step A2C.")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    num_episodes = args.num_episodes
    lr = args.lr
    critic_lr = args.critic_lr
    n = args.n
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')

    # Create the model.
    model = make_actor_model(env.observation_space.shape, env.action_space.n)
    model.summary()

    critic_model = make_critic_model(env.observation_space.shape)
    critic_model.summary()

    # Train the model using A2C and plot the learning curves.
    rein = A2C(model, lr, critic_model, critic_lr, n=n)
    best_reward = -1000
    test_rewards = []
    for i in range(num_episodes):
        actor_hist, critic_hist = rein.train(env, gamma=0.99)

        # Evaluate model every 100 iterations
        if i % 100 == 0:
            multi_test_rewards = []
            lens = []
            # Accumulate results across 20 evaluations for mean and std
            for j in range(20):
                _, _, rewards = rein.generate_episode(env, testing=True)

                if sum(rewards) > best_reward:
                    best_reward = sum(rewards)
                    print(f"New Best Rewards: {best_reward}")

                multi_test_rewards.append(sum(rewards))
                lens.append(len(rewards))
            test_rewards.append(multi_test_rewards)

            # Render video specific
            if render:
                _, actions, rewards = rein.generate_episode(env, render=True, testing=True)
                print(f"Video Actions: {actions}")
                print(f"Video Reward: {sum(rewards)}")

            # Report progress on stdout
            print(f"(Episode {i})\tMean Reward = {np.mean(multi_test_rewards)}, T = {np.mean(lens)}")
            print(f"\tAll Rewards = {[int(r) for r in multi_test_rewards]}")

            # Update results file as we go incase we want to stop early
            with open(f"rewards_a2c_{n}.json", "w") as f:
                json.dump(test_rewards, f)


if __name__ == '__main__':
    main(sys.argv)
