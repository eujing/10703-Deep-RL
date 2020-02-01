import gym
import deeprl_hw2q2.lake_envs as lake_env
import deeprl_hw2q2.rl as rl
import time

def main():
    # mode = "policy_iter"
    # mode = "policy_iter_async_ordered"
    # mode = "policy_iter_async_randperm"
    # mode = "value_iter"
    # mode = "value_iter_async_ordered"
    # mode = "value_iter_async_randperm"
    mode = "value_iter_async_custom"
    fl_dim = "4x4"
    # fl_dim = "8x8"

    env = rl.env_wrapper(f"Deterministic-{fl_dim}-FrozenLake-v0")

    start = time.time()
    # Policy Iteration
    if mode == "policy_iter":
        policy, value_func, improve_iters, eval_iters = rl.policy_iteration_sync(
            env, 0.9, tol=1e-3)
        end = time.time()
        print(f"Num. Improves: {improve_iters}, Num. Evals: {eval_iters}")

    elif mode == "policy_iter_async_ordered":
        policy, value_func, improve_iters, eval_iters = rl.policy_iteration_async_ordered(
            env, 0.9, tol=1e-3)
        end = time.time()
        print(f"Num. Improves: {improve_iters}, Num. Evals: {eval_iters}")

    elif mode == "policy_iter_async_randperm":
        sum_improve_iters = 0
        sum_eval_iters = 0
        for _ in range(10):
            policy, value_func, improve_iters, eval_iters = rl.policy_iteration_async_randperm(
                env, 0.9, tol=1e-3)
            sum_improve_iters += improve_iters
            sum_eval_iters += eval_iters
        end = time.time()
        print(f"Avg Num. Improves: {sum_improve_iters / 10}, Avg. Num. Evals: {sum_eval_iters / 10}")

    # Value Iteration
    elif mode == "value_iter":
        value_func, num_iters = rl.value_iteration_sync(env, 0.9, tol=1e-3)
        policy = rl.value_function_to_policy(env, 0.9, value_func)
        end = time.time()
        print(f"Num Iterations: {num_iters}")

    elif mode == "value_iter_async_ordered":
        value_func, num_iters = rl.value_iteration_async_ordered(env, 0.9, tol=1e-3)
        policy = rl.value_function_to_policy(env, 0.9, value_func)
        end = time.time()
        print(f"Num Iterations: {num_iters}")

    elif mode == "value_iter_async_randperm":
        sum_iters = 0
        for _ in range(10):
            value_func, num_iters = rl.value_iteration_async_randperm(env, 0.9, tol=1e-3)
            policy = rl.value_function_to_policy(env, 0.9, value_func)
            sum_iters += num_iters
        end = time.time()
        print(f"Avg. Num Iterations: {sum_iters / 10}")

    elif mode == "value_iter_async_custom":
        value_func, num_iters = rl.value_iteration_async_custom(env, 0.9, tol=1e-3)
        policy = rl.value_function_to_policy(env, 0.9, value_func)
        end = time.time()
        print(f"Num Iterations: {num_iters}")

    else:
        print("Invalid mode!")
        exit()

    print(f"Time taken: {end - start:.4}s")
    rl.display_policy_letters(env, policy)
    rl.value_func_heatmap(env, value_func, f"{mode}_{fl_dim}")

if __name__ == "__main__":
    main()
