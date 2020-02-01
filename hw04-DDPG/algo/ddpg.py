import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pdb

from .ReplayBuffer import ReplayBuffer
from .ActorNetwork import ActorNetwork
from .CriticNetwork import CriticNetwork

BUFFER_SIZE = 1000000
BATCH_SIZE = 1024
GAMMA = 0.98                    # Discount for rewards.
TAU = 0.05                      # Target network update rate.
ACTOR_HIDDEN = [400, 400]
LEARNING_RATE_ACTOR = 0.0001
CRITIC_HIDDEN = [400, 400]
LEARNING_RATE_CRITIC = 0.001
EPSILON = 0.2
EPSILON_DECAY = 1
STD = 0.01

class EpsilonNormalActionNoise(object):
    """A class for adding noise to the actions for exploration."""

    def __init__(self, mu, sigma, epsilon, decay=1):
        """Initialize the class.

        Args:
            mu: (float) mean of the noise (probably 0).
            sigma: (float) std dev of the noise.
            epsilon: (float) probability in range [0, 1] with
            which to add noise.
        """
        self.mu = mu
        self.sigma = sigma
        self.epsilon = epsilon
        self._decay = decay

    def decay(self):
        self.epsilon *= self._decay

    def __call__(self, action):
        """With probability epsilon, adds random noise to the action.
        Args:
            action: a batched tensor storing the action.
        Returns:
            noisy_action: a batched tensor storing the action.
        """
        if np.random.uniform() > self.epsilon:
            # Our actual action gets clipped too, so make sure this is reflected in replay memory
            return np.clip(action + np.random.normal(self.mu, self.sigma), -1.0, 1.0)
        else:
            return np.random.uniform(-1.0, 1.0, size=action.shape)


class DDPG(object):
    """A class for running the DDPG algorithm."""

    def __init__(self, env, train_outfile_name, test_outfile_name, eval_N):
        """Initialize the DDPG object.

        Args:
            env: an instance of gym.Env on which we aim to learn a policy.
            outfile_name: (str) name of the output filename.
        """
        action_dim = len(env.action_space.low)
        state_dim = len(env.observation_space.low)
        np.random.seed(1337)
        self.env = env
        self.test_outfile = test_outfile_name
        self.train_outfile = train_outfile_name
        self.eval_N = eval_N

        self.eps_norm = EpsilonNormalActionNoise(0, STD, EPSILON, EPSILON_DECAY)
        self.buffer = ReplayBuffer(BUFFER_SIZE)
        self.actor = ActorNetwork(
            None, state_dim, action_dim, BATCH_SIZE, TAU, LEARNING_RATE_ACTOR,
            hidden1=ACTOR_HIDDEN[0], hidden2=ACTOR_HIDDEN[1])
        self.critic = CriticNetwork(
            None, state_dim, action_dim, BATCH_SIZE, TAU, LEARNING_RATE_CRITIC,
            hidden1=CRITIC_HIDDEN[0], hidden2=CRITIC_HIDDEN[1])

    def evaluate(self, num_episodes, noise=None):
        """Evaluate the policy. Noise is not added during evaluation.

        Args:
            num_episodes: (int) number of evaluation episodes.
        Returns:
            success_rate: (float) fraction of episodes that were successful.
            average_return: (float) Average cumulative return.
        """
        test_rewards = []
        success_vec = []
        plt.figure(figsize=(12, 12))
        for i in range(num_episodes):
            s_vec = []
            state = self.env.reset()
            s_t = np.array(state)
            total_reward = 0.0
            done = False
            step = 0
            success = False
            while not done:
                s_vec.append(s_t)
                a_t = self.actor.model.predict(s_t[None])[0]
                if noise is not None:
                    a_t = noise(a_t)
                new_s, r_t, done, info = self.env.step(a_t)
                if done and "goal" in info["done"]:
                    success = True
                new_s = np.array(new_s)
                total_reward += r_t
                s_t = new_s
                step += 1
            success_vec.append(success)
            test_rewards.append(total_reward)
            # if i < 9:
            #     plt.subplot(3, 3, i+1)
            #     s_vec = np.array(s_vec)
            #     pusher_vec = s_vec[:, :2]
            #     puck_vec = s_vec[:, 2:4]
            #     goal_vec = s_vec[:, 4:]
            #     plt.plot(pusher_vec[:, 0], pusher_vec[:, 1], '-o', label='pusher')
            #     plt.plot(puck_vec[:, 0], puck_vec[:, 1], '-o', label='puck')
            #     plt.plot(goal_vec[:, 0], goal_vec[:, 1], '*', label='goal', markersize=10)
            #     plt.plot([0, 5, 5, 0, 0], [0, 0, 5, 5, 0], 'k-', linewidth=3)
            #     plt.fill_between([-1, 6], [-1, -1], [6, 6], alpha=0.1,
            #                      color='g' if success else 'r')
            #     plt.xlim([-1, 6])
            #     plt.ylim([-1, 6])
            #     if i == 0:
            #         plt.legend(loc='lower left', fontsize=28, ncol=3, bbox_to_anchor=(0.1, 1.0))
            #     if i == 8:
            #         # Comment out the line below to disable plotting.
            #         plt.show()
        return np.mean(success_vec), np.mean(test_rewards), np.std(test_rewards)

    def train(self, num_episodes, hindsight=False):
        """Runs the DDPG algorithm.

        Args:
            num_episodes: (int) Number of training episodes.
            hindsight: (bool) Whether to use HER.
        """

        self.buffer.erase()
        # TODO: Burn in buffer?
        print("Burning in bufer...")
        while self.buffer.count() < BUFFER_SIZE / 10:
            state = self.env.reset().astype("float32")
            done = False
            while not done:
                # Collect one episode of experience, saving the states and actions
                # to store_states and store_actions, respectively.
                action = np.random.uniform(-1.0, 1.0, size=(2, )).astype("float32")

                next_state, reward, done, info = self.env.step(action)
                next_state = next_state.astype("float32")

                self.buffer.add(state, action, reward, next_state, done)

                state = next_state

        for i in tf.range(num_episodes):
            state = self.env.reset().astype("float32")
            total_reward = 0.0
            done = False
            step = 0
            TD_loss = 0
            critic_loss = 0
            actor_loss = 0

            store_states = []
            store_actions = []
            while not done:
                # Collect one episode of experience, saving the states and actions
                # to store_states and store_actions, respectively.
                action = self.actor.model(state.reshape(1, -1), training=False)[0]
                noisy_action = self.eps_norm(action)

                # Make sure to copy as hindsight modifies in-place
                store_states.append(state.copy())
                store_actions.append(noisy_action.copy())

                next_state, reward, done, info = self.env.step(noisy_action)
                next_state = next_state.astype("float32")
                total_reward += reward
                step += 1

                # 1) Store in replay buffer
                exp = (state, noisy_action, reward, next_state, done)
                self.buffer.add(state, noisy_action, reward, next_state, done)
                state = next_state

                # 2) Sample minibatch from replay buffer
                experiences = self.buffer.get_batch(BATCH_SIZE)
                states = np.array([exp[0] for exp in experiences], dtype="float32")
                actions = np.array([exp[1] for exp in experiences], dtype="float32")
                rewards = np.array([exp[2] for exp in experiences], dtype="float32")
                next_states = np.array([exp[3] for exp in experiences], dtype="float32")
                dones = np.array([exp[4] for exp in experiences], dtype="int8")

                # 3) Calculate y_i from target networks
                next_actions = self.actor.target(next_states, training=False).numpy()
                y = rewards + \
                    (1 - dones) * GAMMA * self.critic.target([next_states, next_actions], training=False).numpy().squeeze()

                # 4) Update critic using y_i
                # losses, TD_losses = self.critic.train(states, actions, y, dbg=i % 20 == 0 and i >= 100)
                critic_losses, TD_losses = self.critic.train(states, actions, y)
                critic_loss += tf.reduce_mean(critic_losses)
                TD_loss += tf.reduce_mean(TD_losses)

                # 5) Update actor using s_i
                # actor_losses = self.actor.train(states, self.critic.model, dbg=i%20 == 0 and i >= 100)
                actor_losses = self.actor.train(states, self.critic.model)
                actor_loss += actor_losses

                # 6) Update target networks
                self.critic.update_target()
                self.actor.update_target()

            if hindsight:
                # For HER, we also want to save the final next_state.
                store_states.append(next_state)
                self.add_hindsight_replay_experience(store_states,
                                                     store_actions)
            del store_states, store_actions
            store_states, store_actions = [], []

            # Logging
            print(f"Episode {i}: Total reward = {total_reward}")
            TD_loss /= step
            critic_loss /= step
            actor_loss /= step
            print(f"\tTD loss = {TD_loss:.2f} Critic Loss = {critic_loss:.2f} Actor Loss = {actor_loss:.2f}")
            print(f"\tSteps = {step}; Info = {info['done']}")
            self.eps_norm.decay()

            # Record train metrics
            with open(self.train_outfile, "a") as f:
                f.write(f"{total_reward}, {TD_loss}, {critic_loss}, {actor_loss}\n")

            # Record test metrics
            if i % 100 == 0:
                # successes, mean_rewards = self.evaluate(11, noise=self.eps_norm)
                successes, mean_rewards, std_rewards = self.evaluate(self.eval_N)
                print(f"Evaluation: success = {successes:.2f}; return = {mean_rewards:.2f}")
                with open(self.test_outfile, "a") as f:
                    f.write(f"{successes}, {mean_rewards}, {std_rewards}\n")


    def add_hindsight_replay_experience(self, states, actions):
        """Relabels a trajectory using HER.

        Args:
            states: a list of states.
            actions: a list of states.
        """
        her_states, her_rewards = self.env.apply_hindsight(states)

        for i in range(len(actions)):
            s_t = her_states[i]
            a_t = actions[i]
            r_t = her_rewards[i]
            s_tn = her_states[i+1]
            done = (i == len(actions) - 1)
            self.buffer.add(s_t, a_t, r_t, s_tn, done)
