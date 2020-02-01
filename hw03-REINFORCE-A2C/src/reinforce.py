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

def make_model(input_size, output_size):
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


class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, model, lr):
        self.model = model
        opt = keras.optimizers.Adam(lr=lr)
        self.model.compile(optimizer=opt, loss="categorical_crossentropy")


    def policy(self, state):
        return self.model.predict(state.reshape((1, -1)))[0]


    def train(self, env, gamma=1.0, min_batch_size=200):
        # Trains the model on a single episode using REINFORCE.

        all_states, all_actions, all_rewards = [], [], []
        all_G = []

        # Potentially collect across more than 1 episode so we can maintain a minimum batch size
        while len(all_states) < min_batch_size:
            # Collect exactly 1 episode
            states, actions, rewards = self.generate_episode(env)
            all_states.extend(states)
            all_actions.extend(actions)
            all_rewards.extend(rewards)

            # Calculate G by episode, not across a few episodes
            T = len(states)
            G = np.zeros(T)
            G[-1] = rewards[-1]
            for i in reversed(range(T - 1)):
                G[i] = rewards[i] + gamma * G[i + 1]

            # Normalize G, ensure we dont divide by 0
            G = (G - G.mean()) / (G.std() + 0.0001)

            all_G.extend(G)

        states, actions, rewards = np.array(all_states), np.array(all_actions), np.array(all_rewards)
        G = np.array(all_G)

        # Encode G so we can use categorical CE loss
        # Or now, normally OHE actions, but use sample_weight in fit() to reinforce with G
        targets = np.zeros((len(actions), env.action_space.n))
        targets[range(len(actions)), actions] = 1

        history = self.model.fit(states, targets, sample_weight=G, batch_size=1024, epochs=1, verbose=0)
        return history

    def generate_episode(self, env, testing=False, render=False):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        states = []
        actions = []
        rewards = []

        done = False
        curr_state = env.reset()

        while not done:
            action_dist = self.policy(curr_state)
            if testing:
                # Take most likely action
                action = np.random.choice(env.action_space.n, p=action_dist)
            else:
                # Sample action based on softmax probabilities
                action = np.random.choice(env.action_space.n, p=action_dist)

            if render:
                env.render()
            next_state, reward, done, info = env.step(action)

            states.append(curr_state)
            actions.append(action)
            rewards.append(reward)

            curr_state = next_state
        env.close()

        return states, actions, rewards


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The learning rate.")

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
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')

    # Create the model.
    model = make_model(env.observation_space.shape, env.action_space.n)
    model.summary()

    # Train the model using REINFORCE and plot the learning curve.
    rein = Reinforce(model, lr)
    best_reward = -1000
    test_rewards = []
    for i in range(num_episodes):
        hist = rein.train(env, gamma=0.99)

        # Evaluate model every 100 iterations
        if i % 100 == 0:
            multi_test_rewards = []
            lens = []
            # Accumulate results across 20 evaluations for mean and std
            for j in range(20):
                _, _, rewards = rein.generate_episode(env, testing=True)
                # _, _, rewards = rein.generate_episode(env, testing=False)

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
            with open("rewards_reinforce.json", "w") as f:
                json.dump(test_rewards, f)

if __name__ == '__main__':
    main(sys.argv)
