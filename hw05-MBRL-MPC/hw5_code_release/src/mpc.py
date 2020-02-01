import os
import tensorflow as tf
import numpy as np
import gym
import copy
import pdb


class MPC:
    def __init__(self, env, plan_horizon, model, popsize, num_elites, max_iters,
                 num_particles=6,
                 use_gt_dynamics=True,
                 use_mpc=True,
                 use_random_optimizer=False):
        """

        :param env:
        :param plan_horizon:
        :param model: The learned dynamics model to use, which can be None if use_gt_dynamics is True
        :param popsize: Population size
        :param num_elites: CEM parameter
        :param max_iters: CEM parameter
        :param num_particles: Number of trajectories for TS1
        :param use_gt_dynamics: Whether to use the ground truth dynamics from the environment
        :param use_mpc: Whether to use only the first action of a planned trajectory
        :param use_random_optimizer: Whether to use CEM or take random actions
        """
        self.env = env
        self.use_gt_dynamics, self.use_mpc, self.use_random_optimizer = use_gt_dynamics, use_mpc, use_random_optimizer
        self.num_particles = num_particles
        self.plan_horizon = plan_horizon
        self.num_nets = None if model is None else model.num_nets

        self.state_dim, self.action_dim = 8, env.action_space.shape[0]
        self.ac_ub, self.ac_lb = env.action_space.high, env.action_space.low

        # Set up optimizer
        self.model = model

        if use_gt_dynamics:
            self.predict_next_state = self.predict_next_state_gt
            assert num_particles == 1
        else:
            self.predict_next_state = self.predict_next_state_model

        # TODO: write your code here
        # Initialize your planner with the relevant arguments.
        # Write different optimizers for cem and random actions respectively
        self.mu = np.zeros((self.plan_horizon, self.action_dim))
        self.pop_size = popsize
        self.n_elites = num_elites
        self.n_iters = max_iters
        self.actions = None
        self.goal = None
        self.actions = []


    def obs_cost_fn(self, state):
        """ Cost function of the current state """
        # Weights for different terms
        W_PUSHER = 1
        W_GOAL = 2
        W_DIFF = 5

        pusher_x, pusher_y = state[0], state[1]
        box_x, box_y = state[2], state[3]
        goal_x, goal_y = self.goal[0], self.goal[1]

        pusher_box = np.array([box_x - pusher_x, box_y - pusher_y])
        box_goal = np.array([goal_x - box_x, goal_y - box_y])
        d_box = np.sqrt(np.dot(pusher_box, pusher_box))
        d_goal = np.sqrt(np.dot(box_goal, box_goal))
        diff_coord = np.abs(box_x / box_y - goal_x / goal_y)
        # the -0.4 is to adjust for the radius of the box and pusher
        return W_PUSHER * np.max(d_box - 0.4, 0) + W_GOAL * d_goal + W_DIFF * diff_coord

    def predict_next_state_model(self, states, actions):
        """ Given a list of state action pairs, use the learned model to predict the next state"""
        # Randomly use one of the networks
        model = self.model.models[np.random.randint(0, self.num_nets)]

        # Prepare input for the network
        input = np.concatenate((states.astype(np.float32), actions.astype(np.float32)), axis=1)

        # Format output of the network
        output = model(input)
        mean, logvar = self.model.get_output(output)

        # Next states are sampled from normal dist.
        stds = np.exp(0.5 * logvar)
        next_states = np.random.normal(mean, stds)

        assert states.shape == next_states.shape
        return next_states

    def predict_next_state_gt(self, states, actions):
        """ Given a list of state action pairs, use the ground truth dynamics to predict the next state"""
        next_states = []
        for state, action in zip(states, actions):
            next_state = self.env.get_nxt_state(state, action)
            next_states.append(next_state)
        next_states = np.array(next_states)

        return next_states

    def train(self, obs_trajs, acs_trajs, rews_trajs, epochs=5):
        """
        Take the input obs, acs, rews and append to existing transitions the train model.
        Arguments:
          obs_trajs: states
          acs_trajs: actions
          rews_trajs: rewards (NOTE: this may not be used)
          epochs: number of epochs to train for
        """
        all_inputs = []
        all_targets = []
        for obs_traj, acs_traj in zip(obs_trajs, acs_trajs):
            assert len(obs_traj) == len(acs_traj) + 1
            # inputs are pairs of (s, a) combined
            # Remember to leave out goal from states
            inputs = [np.concatenate((s[:-2], a), axis=0)
                    for s, a in zip(obs_traj[:-1], acs_traj)]
            # targets are s' from doing a from s
            targets = [s[:-2] for s in obs_traj[1:]]

            assert len(inputs) == len(targets)
            all_inputs.extend(inputs)
            all_targets.extend(targets)

        all_inputs = np.array(all_inputs)
        all_targets = np.array(all_targets)
        assert all_inputs.shape[1:] == (self.state_dim + self.action_dim, )
        assert all_targets.shape[1:] == (self.state_dim, )
        assert np.all(all_inputs[1:, :self.state_dim] == all_targets[:-1, :])
        return self.model.train(all_inputs, all_targets, epochs=epochs, verbose=False)


    def reset(self):
        self.goal = None
        self.actions = []
        self.mu = np.zeros((self.plan_horizon, self.action_dim))
        self.sigma = 0.5 * np.ones((self.plan_horizon, self.action_dim))

    def act(self, state, t):
        """
        Use model predictive control to find the action give current state.

        Arguments:
          state: current state
          t: current timestep
        """
        assert state.shape == (self.state_dim + 2, )

        N = self.pop_size * self.n_iters

        # MPC
        if self.use_mpc:
            if self.use_random_optimizer:
                planned_actions = self.random(state, N)
                action = planned_actions[0, :]
            else:
                sigma = 0.5 * np.ones((self.plan_horizon, self.action_dim))
                mu = self.CEM(state, self.pop_size, self.n_elites, self.n_iters, self.mu, sigma)
                action = mu[0, :]
                next_mu = np.concatenate((mu[1:, :], np.zeros((1, self.action_dim))), axis=0)
                self.mu = next_mu
            return action
        # Non-MPC
        else:
            if t % self.plan_horizon == 0:
                if self.use_random_optimizer:
                    self.actions = self.random(state, N)
                else:
                    sigma = 0.5 * np.ones((self.plan_horizon, self.action_dim))
                    self.actions = self.CEM(state, self.pop_size, self.n_elites, self.n_iters, self.mu, sigma)
            return self.actions[t % self.plan_horizon]

    def random(self, curr_state, N):
        assert curr_state.shape == (self.state_dim + 2, )
        mu = np.zeros((self.plan_horizon, self.action_dim))
        sigma = 0.5 * np.ones((self.plan_horizon, self.action_dim))

        # Initialize states
        states = []
        init_state = curr_state
        self.goal = init_state[-2:]
        states.append(np.array([init_state[:-2] for _ in range(N)]))

        # Initialize actions
        actions = np.random.normal(
                mu, sigma, size=(N, self.plan_horizon, self.action_dim))
        assert actions.shape == (N, self.plan_horizon, self.action_dim)

        # Initialize costs
        costs = np.zeros(N)
        for m in range(N):
            costs[m] += self.obs_cost_fn(states[-1][m])

        for t in range(self.plan_horizon):
            assert states[-1].shape == (N, self.state_dim)
            assert actions[:, t, :].shape == (N, self.action_dim)
            next_states = self.predict_next_state(states[-1], actions[:, t, :])
            assert next_states.shape == (N, self.state_dim)
            states.append(next_states)

            for m in range(N):
                costs[m] += self.obs_cost_fn(next_states[m])

        states = np.array(states)
        assert states.shape == (self.plan_horizon + 1, N, self.state_dim)

        best_idx = np.argmin(costs)
        best_actions = actions[best_idx, :, :]
        assert best_actions.shape == (self.plan_horizon, self.action_dim)

        return best_actions


    def CEM(self, curr_state, pop_size, n_elites, n_iters, mu, sigma):
        # Verify shapes
        assert curr_state.shape == (self.state_dim + 2, )
        assert mu.shape == (self.plan_horizon, self.action_dim)
        assert sigma.shape == (self.plan_horizon, self.action_dim)

        # Init states
        self.goal = curr_state[-2:]
        init_states = np.zeros((pop_size, self.num_particles, self.state_dim))
        init_states[:, :, :] = curr_state[:-2]

        for i in range(n_iters):
            # Final shape should be (plan_horizon, pop_size, num_particles, state_dim)
            states = [init_states]

            # Init actions
            actions = np.random.normal(mu, sigma, size=(pop_size, self.plan_horizon, self.action_dim))
            assert actions.shape == (pop_size, self.plan_horizon, self.action_dim)

            # Init costs
            costs = np.zeros((pop_size, self.num_particles))
            for m in range(pop_size):
                # Since all initial states are same, pick the first particle's
                costs[m, :] += self.obs_cost_fn(states[-1][m, 0, :])

            # Rollout trajectories
            for t in range(self.plan_horizon):
                next_states = np.zeros((pop_size, self.num_particles, self.state_dim))
                for p in range(self.num_particles):
                    curr_states = states[t][:, p, :]
                    assert curr_states.shape == (pop_size, self.state_dim)
                    p_states = self.predict_next_state(curr_states, actions[:, t, :])
                    assert p_states.shape == (pop_size, self.state_dim)
                    next_states[:, p, :] = p_states
                states.append(next_states)

                # Costs of each member of population
                for m in range(pop_size):
                    for p in range(self.num_particles):
                        costs[m, p] += self.obs_cost_fn(next_states[m, p, :])

            states = np.array(states)
            assert states.shape == (self.plan_horizon + 1, pop_size, self.num_particles, self.state_dim)

            # Update mu and sigma based on top n_elites costs (avg. over particles)
            costs = np.mean(costs, axis=1)
            elite_m = np.argsort(costs)[:n_elites]
            elite_actions = actions[elite_m, :, :]
            assert elite_actions.shape == (n_elites, self.plan_horizon, self.action_dim)

            mu = np.mean(elite_actions, axis=0)
            # Lower bound sigma to prevent a collapse in exploration
            pre_sigma = np.std(elite_actions, axis=0)
            sigma = np.clip(pre_sigma, 0.1, None)
            assert mu.shape == (self.plan_horizon, self.action_dim)
            assert sigma.shape == (self.plan_horizon, self.action_dim)

        return mu
