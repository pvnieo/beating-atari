# stdlib
import sys
# 3p
import gym
from gym.error import UnregisteredEnv
# project
from .gym_env import GymEnv


def create_env(args):
    try:
        gym_env = gym.make(args.env)
    except UnregisteredEnv:
        print(f"Env {args.env} not found in the Gym library. Please try again with a valid environment.")
        sys.exit(1)

    env = GymEnv(gym_env, terminal_on_life_loss=not args.no_terminal_loss, noop_max=args.no_op_max, frame_skip=args.frame_skip,
                 num_stack=args.history_length, screen_size=(args.screen_width, args.screen_height),
                 grayscale=not args.no_gray, clip_reward=not args.no_clip_reward)
    return env
