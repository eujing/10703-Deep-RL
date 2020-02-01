import gym
import envs
import argparse
from algo.ddpg import DDPG

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30000,
                        help="Number of episodes")
    parser.add_argument("--evalN", type=int, default=10,
                        help="Number of times to evaluate")
    parser.add_argument("--hindsight", action="store_true", default=False,
                        help="Use HER")
    parser.add_argument("--testout",
                        help="Test Results output")
    parser.add_argument("--trainout",
                        help="Train Results output")
    return parser.parse_args()

def main():
    args = parse_args()
    env = gym.make('Pushing2D-v0')

    algo = DDPG(env, args.trainout, args.testout, args.evalN)
    algo.train(args.n, hindsight=args.hindsight)


if __name__ == '__main__':
    main()
