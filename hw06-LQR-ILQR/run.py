import gym
from deeprl_hw6.arm_env import TwoLinkArmEnv
from controllers import calc_lqr_input
from ilqr import calc_ilqr_input
import json

USE_ILQR = True
USE_ILQR = False
results_fname = "ilqr_results.json" if USE_ILQR else "lqr_results.json"

if __name__ == "__main__":
    env = gym.make("TwoLinkArm-v0")
    sim_env = gym.make("TwoLinkArm-v0")

    cum_rewards = 0
    done = False
    state = env.reset()
    _ = sim_env.reset()
    actions = None

    actions_log = []
    states_log = []
    costs_log = []

    step = 0
    while not done:
        env.render()
        if not USE_ILQR:
            action = calc_lqr_input(env, sim_env)
        else:
            if actions is None or step % len(actions) == 0:
                actions, costs = calc_ilqr_input(env, sim_env, tN=100)
                print(f"Actions: {actions}")
                costs_log.append(costs)
            action = actions[step % len(actions)]

        # print(f"Action: {action}")
        states_log.append(env.state.tolist())
        actions_log.append(action.tolist())

        next_state, reward, done, _ = env.step(action)
        print(f"Step ({step}) Reward: {reward}")
        cum_rewards += reward
        state = next_state
        step += 1

    print(f"Total rewards = {cum_rewards}")
    print(f"Steps = {step}")
    with open(results_fname, "w") as f:
        json.dump({
            "states": states_log,
            "actions": actions_log,
            "costs": costs_log
        }, f)

    env.render(close = True)
    env.close()

