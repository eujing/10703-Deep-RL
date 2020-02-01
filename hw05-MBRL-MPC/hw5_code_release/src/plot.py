import matplotlib.pyplot as plt
import json
import numpy as np
import pdb

with open("single_dynamics_warmup.json", "r") as f:
    results = json.load(f)

with open("mbrl_pets.json", "r") as f:
    results = json.load(f)

# Single Losses and RMSEs
# losses = results["losses"][0]
# rmses = results["rmses"][0]
# iters = range(len(losses))

# fig, axs = plt.subplots(2, 1)

# axs[0].plot(iters, losses, c="C0")
# axs[0].set_xlabel("Iterations")
# axs[0].set_ylabel("Loss")
# axs[0].set_title("Loss over iterations (Warmup)")

# axs[1].plot(iters, rmses, c="C1")
# axs[1].set_xlabel("Iterations")
# axs[1].set_ylabel("RMSE")
# axs[1].set_title("RMSE over iterations (Warmup)")

# plt.show()

# PET Losses and RMSEs
# net1_losses = sum([loss[0] for loss in results["losses"]], [])
# net2_losses = sum([loss[1] for loss in results["losses"]], [])
# net1_rmses = sum([rmse[0] for rmse in results["rmses"]], [])
# net2_rmses = sum([rmse[1] for rmse in results["rmses"]], [])
# iters = range(len(net1_losses))

# fig, axs = plt.subplots(2, 1)

# axs[0].plot(iters, net1_losses, alpha=0.5, label="Net 1")
# axs[0].plot(iters, net2_losses, alpha=0.5, label="Net 2")
# axs[0].set_xlabel("Iterations (5 Training Steps Each Epoch)")
# axs[0].set_ylabel("Loss")
# axs[0].set_title("Losses over Iterations")
# axs[0].legend()

# axs[1].plot(iters, net1_rmses, alpha=0.5, label="Net 1")
# axs[1].plot(iters, net2_rmses, alpha=0.5, label="Net 2")
# axs[1].set_xlabel("Iterations (5 Training Steps Each Epoch)")
# axs[1].set_ylabel("RMSE")
# axs[1].set_title("RMSE over Iterations")
# axs[1].legend()

# plt.show()

# PET Success Rates
cem_succ = results["cem_succ_rates"]
rand_succ = results["rand_succ_rates"]
iters = range(len(cem_succ))

plt.figure(figsize=(4, 2))
plt.plot(iters, cem_succ, label="CEM")
plt.plot(iters, rand_succ, label="Random")
plt.xlabel("Iterations (every 50)", fontsize=26)
plt.ylabel("Success Rate", fontsize=26)
plt.title("Evaluated Success Rates Over Training", fontsize=30)
plt.legend(fontsize=22)
plt.show()
