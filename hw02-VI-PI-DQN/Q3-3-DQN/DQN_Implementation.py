#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse
import keras.models as models
import keras.layers as layers
import keras.optimizers as optimizers
import keras.losses as losses
import keras.backend as K
import collections
import random
import os
import json
import matplotlib.pyplot as plt

# Don't run on GPU, its slower
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

class QNetwork():
    # This class essentially defines the network architecture. 
    # The network should take in state of the world as an input, 
    # and output Q values of the actions available to the agent as the output. 

    def __init__(self, environment_name, hidden_units, learning_rate):
        # Define your network architecture here. It is also a good idea to define any training operations 
        # and optimizers here, initialize your variables, or alternately compile your model here.  

        self.hidden_units = hidden_units
        self.env = gym.make(environment_name)

        # Determined from env
        self.input_shape = self.env.observation_space.shape
        self.output_dim = self.env.action_space.n

        # Model Architecture
        self.model = models.Sequential([
            layers.Dense(self.hidden_units, activation="relu",
                         input_shape=self.input_shape),
            layers.Dropout(0.5),
            layers.Dense(self.hidden_units, activation="relu"),
            # layers.Dropout(0.1),
            layers.Dense(self.output_dim, activation="linear"),
            ])

        # Set up optimizer
        opt = optimizers.Adam(lr=learning_rate)

        # Set up loss, just MSE between y_j and specific Q value
        self.model.compile(optimizer=opt, loss="mse")

    def reset_weights(self):
        for layer in self.model.layers:
            if hasattr(layer, "kernel_initializer"):
                layer.kernel.initializer.run()

    def save_model_weights(self, suffix):
        # Helper function to save your model / weights. 
        self.model.save_weights(f"./train_weights_{suffix}.h5")

    def load_model(self, suffix):
        # Helper function to load an existing model.
        # e.g.: torch.save(self.model.state_dict(), model_file)
        # self.model.load_weights(model, f"model_weights_{suffix}.h5")
        # self.train_model.load_weights(model, f"model_weights_{suffix}.h5")
        pass

    def load_model_weights(self, fname):
        # Helper funciton to load model weights. 
        # e.g.: self.model.load_state_dict(torch.load(model_file))
        self.model.load_weights(fname)


class Replay_Memory():

    def __init__(self, memory_size=50000, burn_in=10000):

        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the 
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions. 

        # Hint: you might find this useful:
        #       collections.deque(maxlen=memory_size)
        self.memory = collections.deque(maxlen=memory_size)
        self.burn_in = burn_in

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
        # You will feed this to your model to train.

        return random.sample(self.memory, batch_size)

    def append(self, transition):
        # Appends transition to the memory.
        self.memory.appendleft(transition)


class DQN_Agent():

    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #   (a) Epsilon Greedy Policy.
    #   (b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, environment_name, hidden_units, learning_rate,
                 num_episodes, epsilon_gen, gamma,
                 batch_size=32, memory_size=int(5e4), burn_in=1e4,
                 test_every=100, render=False):
        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.
        self.environment_name = environment_name
        self.env = gym.make(environment_name)
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.qnet = QNetwork(environment_name, hidden_units, learning_rate)
        self.replay_buffer = Replay_Memory(memory_size, burn_in=burn_in)
        self.num_episodes = num_episodes
        self.epsilon_gen = epsilon_gen
        self.gamma = gamma
        self.batch_size = batch_size
        self.test_every = test_every
        self.render = render


    def epsilon_greedy_policy(self, state, epsilon):
        # Creating epsilon greedy probabilities to sample from.
        if random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            return self.greedy_policy(state)

    def greedy_policy(self, state):
        # Creating greedy policy for test time.
        q_values = self.qnet.model.predict(state.reshape((1, -1)))
        return np.argmax(q_values)

    def train(self):
        # In this function, we will train our network.
        # If training without experience replay_memory, then you will interact with the environment
        # in this function, while also updating your network parameters.

        # When use replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.
        self.burn_in_memory()
        print("Done burning in memory!")

        train_rewards = []
        test_rewards = []
        test_TD_errors = []

        third = self.num_episodes // 3

        mtn_car = self.environment_name == "MountainCar-v0"
        if mtn_car:
            last_succeeded = 0
            best_x = -1.2

        fname = f"{self.environment_name}-checkpoint-0"
        print(f"Saved model weights to: train_weights_{fname}.h5!")
        self.qnet.save_model_weights(fname)
        for i in range(1, self.num_episodes + 1):
            curr_state = self.env.reset()

            done = False
            t = 0
            cum_reward = 0.0
            epsilon = next(self.epsilon_gen)

            while not done:
                # Interact with environment
                a = self.epsilon_greedy_policy(curr_state, epsilon)
                next_state, reward, done, info = self.env.step(a)
                cum_reward += reward

                if mtn_car and next_state[0] > best_x:
                    best_x = next_state[0]
                    print(f"(Episode = {i}) Best x so far: {best_x}")
                if mtn_car and next_state[0] >= 0.5:
                    print(f"(Episode = {i}) Reached goal!")

                # Store transition in replay buffer
                transition = (curr_state, a, reward, next_state, done)
                self.replay_buffer.append(transition)

                for _ in range(1):
                    # Sample minibatch from replay buffer
                    hist_transitions = self.replay_buffer.sample_batch(self.batch_size)
                    batch_size = len(hist_transitions)  # Actual batch size
                    hist_next_states = np.array([next_s for (_, _, _, next_s, _) in hist_transitions])

                    hist_next_q_vals = self.qnet.model.predict(hist_next_states)
                    hist_next_max_q = np.max(hist_next_q_vals, axis=1)

                    # DEBUGGING
                    # print("Q of next state")
                    # print(hist_next_q_vals)
                    # print("max Q of next state")
                    # print(hist_next_max_q)

                    # Construct training inputs and output
                    y = np.array([
                        r if done else r + self.gamma * max_q
                        for (_, _, r, next_s, done), max_q in zip(hist_transitions, hist_next_max_q)
                    ])
                    hist_states = np.array([curr_s for (curr_s, _, _, _, _) in hist_transitions])
                    hist_actions = np.array([a for (_, a, _, _, _) in hist_transitions])
                    hist_q_vals = self.qnet.model.predict(hist_states)

                    hist_q_vals[range(batch_size), hist_actions] = y

                    # Train Q Network on minibatch
                    loss = self.qnet.model.fit(hist_states, hist_q_vals, epochs=1, verbose=0)

                # Prepare for next step
                curr_state = next_state
                t += 1

            train_rewards.append(cum_reward)

            # Evaluate on test
            if i % self.test_every == 0:
                rewards, TD_errors = self.test()
                test_rewards.append(rewards)
                test_TD_errors.append(TD_errors)
                print(f"(Episode = {i}) Epsilon = {epsilon:.4f}, Test Avg. Reward = {np.mean(rewards):.2f}, Test Avg. TD Err. = {np.mean(TD_errors):.4f}")

            # Generate video
            if i % third == 0:
                if self.render:
                    test_video(self, self.environment_name, i)
                fname = f"{self.environment_name}-checkpoint-{int(i / third)}"
                print(f"Saved model weights to: train_weights_{fname}.h5!")
                self.qnet.save_model_weights(fname)

        return test_rewards, test_TD_errors


    def test(self, model_suffix=None):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory. 
        if model_suffix is not None:
            self.qnet.load_model(model_suffix)
        rewards = []
        TD_errors = []
        for i in range(20):
            curr_state = self.env.reset()

            done = False
            cum_reward = 0.0
            TD_error = []

            while not done:
                # Interact with environment
                a = self.greedy_policy(curr_state)
                next_state, reward, done, info = self.env.step(a)

                # Accumulate Reward
                cum_reward += reward

                # Accumulate TD Error
                curr_v = self.qnet.model.predict(curr_state.reshape(1, -1))[0, a]
                next_v = np.max(self.qnet.model.predict(next_state.reshape(1, -1)))
                TD_error.append(np.abs(reward + self.gamma * next_v - curr_v))

                curr_state = next_state

            rewards.append(cum_reward)
            TD_errors.append(np.mean(TD_error))

        return rewards, TD_errors

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        curr_state = self.env.reset()
        n = 0

        cum_reward = 0.0
        reached_goal = False
        best_x = -1.2
        while n < self.replay_buffer.burn_in:
            # Random action
            # a = self.env.action_space.sample()

            # Randomish action from freshly initialized network
            a = self.epsilon_greedy_policy(curr_state, 0.5)
            next_state, reward, done, info = self.env.step(a)

            if next_state[0] > best_x:
                best_x = next_state[0]

            transition = (curr_state, a, reward, next_state, done)
            self.replay_buffer.append(transition)
            n += 1
            cum_reward += reward

            if not done:
                curr_state = next_state
            else:
                reached_goal = reached_goal or cum_reward > -200
                curr_state = self.env.reset()


# Note: if you have problems creating video captures on servers without GUI,
#       you could save and relaod model to create videos on your laptop. 
def test_video(agent, env, epi):
    # Usage:
    #   you can pass the arguments within agent.train() as:
    #       if episode % int(self.num_episodes/3) == 0:
    #           test_video(self, self.environment_name, episode)
    save_path = "./videos-%s-%s" % (env, epi)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # To create video
    env = gym.wrappers.Monitor(agent.env, save_path, force=True)
    reward_total = []
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = agent.epsilon_greedy_policy(state, 0.05)
        next_state, reward, done, info = env.step(action)
        state = next_state
        reward_total.append(reward)
    print("Video reward_total: {}".format(np.sum(reward_total)))
    agent.env.close()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str)
    parser.add_argument('--render',dest='render',type=int,default=0)
    parser.add_argument('--train',dest='train',type=int,default=1)
    parser.add_argument('--model',dest='model_file',type=str)
    parser.add_argument('--plot', dest='metrics_file', type=str)
    return parser.parse_args()

def constant_gen(eps):
    while True:
        yield eps

def linear_decay_gen(start, delta, min_val=0):
    value = start
    while True:
        yield value
        value -= delta
        value = max(min_val, value)

def main(args):
    args = parse_arguments()
    environment_name = args.env

    # Setting the session to allow growth, so it doesn't allocate all GPU memory. 
    # gpu_ops = tf.GPUOptions(allow_growth=True)
    # config = tf.ConfigProto(gpu_options=gpu_ops)
    # sess = tf.Session(config=config)

    # Setting this as the default tensorflow session. 
    # keras.backend.tensorflow_backend.set_session(sess)

    # You want to create an instance of the DQN_Agent class here, and then train / test it. 
    render = False

    if environment_name == "CartPole-v0":
        learning_rate = 0.0002
        gamma = 0.99
        epsilon_gen = linear_decay_gen(0.8, 4.5e-5, 0.05)
        episodes = 5000
        hidden_units = 25
        memory_size = int(5e6)
        burn_in = 1e4
        batch_size = 32
    else:
        learning_rate = 0.001
        learning_rate = 0.0005
        gamma = 1
        epsilon_gen = linear_decay_gen(0.5, 4.5e-6, 0.05)
        episodes = 10000
        hidden_units = 80
        memory_size = int(2e4)
        burn_in = 1e4
        batch_size = 128

    dqn_agent = DQN_Agent(environment_name, hidden_units, learning_rate,
                          episodes, epsilon_gen, gamma,
                          render=render, memory_size=memory_size, burn_in=burn_in)
    if args.train == 1:
        test_rewards, test_TD_errors = dqn_agent.train()
        fname = f"{environment_name}-final"
        dqn_agent.qnet.save_model_weights(fname)
        print(f"Saved model weights to: train_weights_{fname}.h5!")

        with open(f"metrics_{environment_name}.json", "w") as f:
            json.dump({"rewards": test_rewards, "TD_errors": test_TD_errors}, f)
        print(f"Saved train metrics to: train_weights_{environment_name}.json!")

    if args.render == 1:
        dqn_agent.qnet.load_model_weights(args.model_file)
        test_video(dqn_agent, dqn_agent.environment_name, "test")

    if args.metrics_file:
        with open(args.metrics_file, "r") as f:
            metrics = json.load(f)

            rewards = np.array(metrics["rewards"])
            TD_errors = np.array(metrics["TD_errors"])

            plt.figure(figsize=(8, 6), dpi=60)
            iters = range(1, len(rewards) + 1)
            plt.plot(iters, rewards.mean(axis=1), label="Mean")
            plt.errorbar(iters, rewards.mean(axis=1), yerr=2 * rewards.std(axis=1),
                         label="Errors across 20 episodes", alpha=0.3)
            plt.legend()
            plt.xlabel("Iterations (per 100)")
            plt.ylabel("Cumulative Episode Reward")
            plt.title("Cumulative Test Rewards across Iterations")
            plt.savefig("cum_rewards")

            plt.figure(figsize=(8, 6), dpi=60)
            iters = range(1, len(TD_errors) + 1)
            plt.plot(iters, TD_errors.mean(axis=1), label="Mean")
            plt.errorbar(iters, TD_errors.mean(axis=1), yerr=2 * TD_errors.std(axis=1),
                         label="Errors across 20 episodes", alpha=0.3)
            plt.legend()
            plt.xlabel("Iterations (per 100)")
            plt.ylabel("Average TD Error per Timestep")
            plt.title("Average TD Error per Timestep across Iterations")
            plt.savefig("avg_TD_errors")

if __name__ == '__main__':
    main(sys.argv)

