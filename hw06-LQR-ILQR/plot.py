import matplotlib.pyplot as plt
import json
import numpy as np
import pdb

ILQR = True
# ILQR = False

if __name__ == "__main__":
    if not ILQR:
        with open("lqr_results.json", "r") as f:
            results = json.load(f)
    else:
        with open("ilqr_results.json", "r") as f:
            results = json.load(f)

    states = np.array(results["states"])
    positions = states[:, :2]
    dpos = np.diff(positions, axis=0)
    velocities = states[:, 2:]
    dvel = np.diff(velocities, axis=0)
    actions = np.array(results["actions"])
    du = np.diff(actions, axis=0)

    fig, axs = plt.subplots(3, 1)
    n = len(positions)
    axs[0].plot(positions[:, 0], positions[:, 1])
    axs[0].arrow(
            positions[n//10, 0], positions[n//10, 1],
            dpos[n//10, 0], dpos[n//10, 1],
            shape='full', lw=0, length_includes_head=True, head_width=.08)
    axs[0].set_xlabel("q[0]")
    axs[0].set_ylabel("q[1]")
    axs[0].set_title("q by Coordinate against Time")

    axs[1].plot(velocities[:, 0], velocities[:, 1])
    if ILQR:
        axs[1].arrow(
                velocities[n//10, 0], velocities[n//10, 1],
                dvel[n//10, 0], dvel[n//10, 1],
                shape='full', lw=0, length_includes_head=True, head_width=3)
    else:
        axs[1].arrow(
                velocities[n//10, 0], velocities[n//10, 1],
                dvel[n//10, 0], dvel[n//10, 1],
                shape='full', lw=0, length_includes_head=True, head_width=20)
    axs[1].set_xlabel("q_dot[0]")
    axs[1].set_ylabel("q_dot[1]")
    axs[1].set_title("q_dot by Coordinate against Time")

    axs[2].plot(actions[:, 0], actions[:, 1])
    if ILQR:
        axs[2].arrow(
                actions[n//10, 0], actions[n//10, 1],
                du[n//10, 0], du[n//10, 1],
                shape='full', lw=0, length_includes_head=True, head_width=30)
    else:
        axs[2].arrow(
                actions[n//10, 0], actions[n//10, 1],
                du[n//10, 0], du[n//10, 1],
                shape='full', lw=0, length_includes_head=True, head_width=300)
    axs[2].set_xlabel("u[0]")
    axs[2].set_ylabel("u[1]")
    axs[2].set_title("Action u by Coordinate against Time")

    plt.tight_layout()
    plt.show()

    if ILQR:
        costs = results["costs"]
        fig, axs = plt.subplots(len(costs), 1)
        for i, c in enumerate(costs):
            axs[i].plot(range(len(c)), c)
            axs[i].set_xlabel("Iteration")
            axs[i].set_ylabel("Cost")
            axs[i].set_title(f"Total Costs across Iterations (iLQR Call {i+1})")

        plt.tight_layout()
        plt.show()
