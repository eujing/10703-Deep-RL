"""LQR, iLQR and MPC."""

import numpy as np
import pdb
from scipy.linalg import solve_continuous_are, inv



def simulate_dynamics(env, x, u, dt=1e-5):
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
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    xdot: np.array
      This is the **CHANGE** in x. i.e. (x[1] - x[0]) / dt
      If you return x you will need to solve a different equation in
      your LQR controller.
    """

    env.state = x.copy()
    x_next, _, _, _ = env.step(u, dt=dt)
    xdot = (x_next - x) / dt

    assert xdot.shape == x.shape
    return xdot


def approximate_A(env, x, u, delta=1e-5, dt=1e-5):
    """Approximate A matrix using finite differences.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 3
      link arm.
    x: np.array
      The state to test. You will need to perturb this.
    u: np.array
      The command to test.
    delta: float
      How much to perturb the state by.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    A: np.array
      The A matrix for the dynamics at state x and command u.
    """
    A = np.zeros((x.shape[0], x.shape[0]))

    for dim in range(x.shape[0]):
        dx = np.zeros_like(x)
        dx[dim] = delta

        x1 = x - dx
        x2 = x + dx

        f1 = simulate_dynamics(env, x1, u, dt=dt)
        f2 = simulate_dynamics(env, x2, u, dt=dt)

        A[:, dim] = (f2 - f1) / (2 * delta)

    assert A.shape == (x.shape[0], x.shape[0])
    return A


def approximate_B(env, x, u, delta=1e-5, dt=1e-5):
    """Approximate B matrix using finite differences.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test.
    u: np.array
      The command to test. You will ned to perturb this.
    delta: float
      How much to perturb the state by.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    B: np.array
      The B matrix for the dynamics at state x and command u.
    """
    B = np.zeros((x.shape[0], u.shape[0]))

    for dim in range(u.shape[0]):
        du = np.zeros_like(u)
        du[dim] = delta

        u1 = u - du
        u2 = u + du

        f1 = simulate_dynamics(env, x, u1, dt=dt)
        f2 = simulate_dynamics(env, x, u2, dt=dt)

        B[:, dim] = (f2 - f1) / (2 * delta)

    assert B.shape == (x.shape[0], u.shape[0])
    return B


def calc_lqr_input(env, sim_env):
    """Calculate the optimal control input for the given state.

    If you are following the API and simulate dynamics is returning
    xdot, then you should use the scipy.linalg.solve_continuous_are
    function to solve the Ricatti equations.

    Parameters
    ----------
    env: gym.core.Env
      This is the true environment you will execute the computed
      commands on. Use this environment to get the Q and R values as
      well as the state.
    sim_env: gym.core.Env
      A copy of the env class. Use this to simulate the dynamics when
      doing finite differences.

    Returns
    -------
    u: np.array
      The command to execute at this point.
    """
    x = env.state.copy()
    x_target = env.goal.copy()
    Q = env.Q
    R = env.R
    u_equi = np.zeros((2, ))

    # Approximate A and B around given state x and
    # u_equi (which gives x = f(x, u_equi)?)
    A = approximate_A(sim_env, x, u_equi)
    B = approximate_B(sim_env, x, u_equi)

    # Solve Ricatti equations for P
    P = solve_continuous_are(A, B, Q, R)
    # Calculate gain function
    K = inv(R) @ B.T @ P
    du = -K @ (x - x_target)
    u = u_equi + du

    assert u.shape == (2, )
    return u
