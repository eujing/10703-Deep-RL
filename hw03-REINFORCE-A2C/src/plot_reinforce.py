import numpy as np
import json
import matplotlib.pyplot as plt

if __name__ == "__main__":
    with open("rewards_reinforce.json", "r") as f:
        rewards = np.array(json.load(f))

    iters = range(len(rewards))
    means = rewards.mean(axis=1)
    stds = rewards.std(axis=1)

    plt.plot(iters, means, label="Mean Reward")
    plt.errorbar(iters, means, yerr=stds, alpha=0.3, label="Error")
    plt.xlabel("Iterations (every 100)")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Rewards against Iterations (every 100)")
    plt.legend(loc="lower right")
    plt.show()
