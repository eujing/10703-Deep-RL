"""LQR, iLQR and MPC."""

from controllers import approximate_A, approximate_B
import numpy as np
from scipy.linalg import pinv
import pdb


def simulate_dynamics_next(env, x, u):
    """Step simulator to see how state changes.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.
    u: np.array
      The command to test. When approximating B you will need to
      perturb this.

    Returns
    -------
    next_x: np.array
    """
    env.state = x.copy()
    x_next, _, _, _ = env.step(u)
    return x_next


def cost_inter(env, x, u):
    """intermediate cost function

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.
    u: np.array
      The command to test. When approximating B you will need to
      perturb this.

    Returns
    -------
    l, l_x, l_xx, l_u, l_uu, l_ux. The first term is the loss, where the remaining terms are derivatives respect to the
    corresponding variables, ex: (1) l_x is the first order derivative d l/d x (2) l_xx is the second order derivative
    d^2 l/d x^2
    """
    m = x.shape[0]
    n = u.shape[0]

    l = u.T @ u
    l_x = np.zeros_like(m)
    l_xx = np.zeros((m, m))
    l_u = 2 * u
    l_uu = 2 * np.eye(n)
    l_ux = np.zeros((n, m))
    return l, l_x, l_xx, l_u, l_uu, l_ux


def cost_final(env, x):
    """cost function of the last step

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.

    Returns
    -------
    l, l_x, l_xx The first term is the loss, where the remaining terms are derivatives respect to the
    corresponding variables
    """
    m = x.shape[0]
    x_diff = x - env.goal
    multiplier = 10**4

    l = multiplier * (x_diff.T @ x_diff)
    l_x = 2 * multiplier * x_diff
    l_xx = 2 * multiplier * np.eye(m)


    assert l.shape == ()
    assert l_x.shape == (m, )
    assert l_xx.shape == (m, m)

    return l, l_x, l_xx


def simulate(env, x0, U):
    states = [x0]

    env.state = x0.copy()
    for u in U:
        x_next, _, _, _ = env.step(u)
        states.append(x_next)

    return np.array(states)


def calc_ilqr_input(env, sim_env, tN=50, max_iter=1e6):
    """Calculate the optimal control input for the given state.


    Parameters
    ----------
    env: gym.core.Env
      This is the true environment you will execute the computed
      commands on. Use this environment to get the Q and R values as
      well as the state.
    sim_env: gym.core.Env
      A copy of the env class. Use this to simulate the dynamics when
      doing finite differences.
    tN: number of control steps you are going to execute
    max_itr: max iterations for optmization

    Returns
    -------
    U: np.array
      The SEQUENCE of commands to execute. The size should be (tN, #parameters)
    """
    x0 = env.state.copy()
    Q = env.Q
    R = env.R
    # U = np.array([env.action_space.sample() for _ in range(tN)])
    U = np.zeros((tN, 2))
    m = x0.shape[0]
    n = U[0].shape[0]
    dt = 1e-3
    cost = 0
    reg = np.eye(n) * 1.0
    costs = []

    for i in range(int(max_iter)):
        # Get state trajectory
        X = simulate(sim_env, x0, U)
        assert U.shape[0] == tN
        assert X.shape[0] == tN + 1

        # Initialize placeholders
        l = np.zeros((tN + 1, ))
        l_x = np.zeros((tN + 1, m))
        l_xx = np.zeros((tN + 1, m, m))
        l_u = np.zeros((tN, n))
        l_uu = np.zeros((tN, n, n))
        l_ux = np.zeros((tN, n, m))
        f_x = np.zeros((tN, m, m))
        f_u = np.zeros((tN, m, n))
        V_x = np.zeros((tN + 1, m))
        V_xx = np.zeros((tN + 1, m, m))
        k = np.zeros((tN, n))
        K = np.zeros((tN, n, m))

        # Calculate all costs and partial derivatives
        for t in range(tN):
            x, u = X[t], U[t]

            l[t], l_x[t, :], l_xx[t, :], l_u[t, :], l_uu[t, :, :], l_ux[t, :, :] = cost_inter(sim_env, x, u)

            # Approximate xdot(t) = A x(t) + B u(t), and x(t+1) = x(t) + xdot(t) * dt
            # So later x(t+1) = x(t) + (A x(t) + B u(t)) * dt
            A = approximate_A(sim_env, x, u)
            B = approximate_B(sim_env, x, u)

            # Dynamics is x(t+1) = f(x(t), u(t))
            # Partial derivatives of f wrt x = I + A * dt
            f_x[t, :, :] = np.eye(m) + A * dt
            # Partial derivatives of f wrt x = 0 + B * dt
            f_u[t, :, :] = B * dt

        l *= dt
        l_x *= dt
        l_xx *= dt
        l_u *= dt
        l_uu *= dt
        l_ux *= dt
        l[tN], l_x[tN, :], l_xx[tN, :, :] = cost_final(sim_env, X[-1])

        # Check for early convergence
        # ===========================
        curr_cost = l.sum()
        costs.append(curr_cost)
        if cost != 0:
            diff_perc = np.abs((curr_cost - cost) / cost)
            # print(f"Iter ({i}): Old Cost: {cost:.2f} Curr Cost: {curr_cost:.2f} Diff Perc: {diff_perc:.4f}")
            if diff_perc < 1e-3:
                print(f"Exiting early at iteration {i}")
                return U, costs
        cost = curr_cost

        # Start Dynamic Programming for Backpass
        # ======================================

        # Initial values from the back
        V_x[tN, :] = l_x[tN, :].copy()
        V_xx[tN, :, :] = l_xx[tN, :, :].copy()

        for t in reversed(range(tN)):
            Q_x = l_x[t] + f_x[t].T @ V_x[t+1]
            Q_u = l_u[t] + f_u[t].T @ V_x[t+1]
            Q_xx = l_xx[t] + f_x[t].T @ V_xx[t+1] @ f_x[t]
            Q_ux = l_ux[t] + f_u[t].T @ V_xx[t+1] @ f_x[t]
            Q_uu = l_uu[t] + f_u[t].T @ V_xx[t+1] @ f_u[t]

            # Safe inverse with regularization
            Q_uu_inv = pinv(Q_uu + reg)
            k[t, :] = -Q_uu_inv @ Q_u
            K[t, :, :] = -Q_uu_inv @ Q_ux

            # Current gradients for value function for prev timestep
            V_x[t] = Q_x - K[t].T @ Q_uu @ k[t]
            V_xx[t] = Q_xx - K[t].T @ Q_uu @ K[t]

        # Forward Pass
        # ============
        updated_U = np.zeros_like(U)
        updated_X = np.zeros_like(X)
        updated_X[0, :] = x0.copy()

        for t in range(tN):
            new_x = updated_X[t]
            new_u = U[t] + K[t] @ (new_x - X[t]) + k[t]
            next_x = simulate_dynamics_next(sim_env, new_x, new_u)

            updated_U[t, :] = new_u
            updated_X[t+1, :] = next_x

        X = updated_X.copy()
        U = updated_U.copy()
        final_l = l.copy()

    return U, costs
