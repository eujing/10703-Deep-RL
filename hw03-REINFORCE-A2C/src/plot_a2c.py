import numpy as np
import json
import matplotlib.pyplot as plt

if __name__ == "__main__":
    N = [1, 20, 50, 100]
    rewards = {}
    for fname, n in [(f"rewards_a2c_{n}.json", n) for n in N]:
        with open(fname, "r") as f:
            rewards[n] = np.array(json.load(f))

    fig, ax = plt.subplots(len(N), 1, sharey=True, figsize=(6, 8))
    for i, n in enumerate(N):
        iters = range(len(rewards[n]))
        means = rewards[n].mean(axis=1)
        stds = rewards[n].std(axis=1)

        ax[i].plot(iters, means, label="Mean Reward")
        ax[i].errorbar(iters, means, yerr=stds, alpha=0.3, label="Error")
        ax[i].set_ylabel("Cumulative Reward")
        ax[i].set_title(f"N = {n}")
        ax[i].legend(loc="lower right")

    plt.setp([a.get_xticklabels() for a in ax[0:-1]], visible=False)
    ax[-1].set_xlabel("Iterations (every 100)")
    fig.suptitle("Cumulative Rewards against Iterations (every 100)")
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig("rewards_a2c")
    plt.show()
