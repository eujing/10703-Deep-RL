import csv
import matplotlib.pyplot as plt

TEST_FILE = "results/ddpg3_test.csv"
TRAIN_FILE = "results/ddpg3_train.csv"

TEST_FILE = "results/HER2_test.csv"
TRAIN_FILE = "results/HER2_train.csv"

if __name__ == "__main__":
    test_results = {
        "success": [],
        "rewards_mean": [],
        "rewards_std": []
    }

    train_results = {
        "success": [],
        "TD_loss": [],
        "critic_loss": [],
        "actor_loss": []
    }

    with open(TEST_FILE, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for succ, mean, std in reader:
            test_results["success"].append(float(succ))
            test_results["rewards_mean"].append(float(mean))
            test_results["rewards_std"].append(float(std))

    with open(TRAIN_FILE, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for succ, TD, critic, actor in reader:
            train_results["success"].append(float(succ))
            train_results["TD_loss"].append(float(TD))
            train_results["critic_loss"].append(float(critic))
            train_results["actor_loss"].append(float(actor))

    train_iters = range(0, len(train_results["success"]))
    test_iters = range(0, len(test_results["success"]) * 100, 100)

    fig, ax1 = plt.subplots()
    ax1.plot(test_iters, test_results["rewards_mean"], label="Mean Cum. Rewards")
    ax1.errorbar(test_iters, test_results["rewards_mean"], yerr=test_results["rewards_std"],
                 alpha=0.3, label="Error")
    ax1.set_ylabel("Cumulative Rewards", fontsize=20)

    # ax2 = ax1.twinx()
    # ax2.plot(train_iters, train_results["actor_loss"], label="Actor Loss",
    #          c="C2", linewidth=0.3)
    # ax2.set_ylabel("Actor Loss", fontsize=20)
    # plt.plot(train_iters, train_results["critic_loss"], label="Critic Loss")
    # plt.plot(train_iters, train_results["TD_loss"], label="TD Loss")
    ax1.set_ylim((-40, 5))
    plt.title("Cumulative Rewards over Iterations", fontsize=24)
    plt.xlabel("Iterations", fontsize=20)
    ax1.legend(fontsize=16, loc="upper right")
    # ax2.legend(fontsize=16, loc="upper left")
    plt.show()


