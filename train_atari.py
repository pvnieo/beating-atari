# stdlib
import sys
import os
# 3p
import torch
from torch.optim import Adam
# project
from options.train_options import TrainOptions
from envs import create_env
from networks import create_network
from agents import create_agent
from utils.epsilon_greedy import AnnealedEpsilonGreedyPolicy
from utils.experience_replay import SimpleExperienceReplay
from utils.logger import Logger

# from agents.dqn import DQNNips, DQN
# from agents.double_dqn import DoubleDQN
# from envs.gym_env import GymEnv
# from networks.simple_fc import FCNetwork
# from networks.dqn_nature import DQNNatureNetwork
# from networks.dqn_nips import DQNNipsNetwork

# # ################### Argument parser will be added later ################### #

# # ### global constant ### #
# OUTPUT_DIR = "outputs"
# ENV_NAME = 'Breakout-v4'
# networkk = "dqn_nature"  # [dqn_nature, dqn_nips]
# agent_name = "dqn_nips"  # [dqn, dqn_nips, double_dqn]

# EXPLORATION_STEPS = 1000  # 1000000
# MEM_MAX_SIZE = 1000000
# TARGET_NETWORK_UPDATE = 10  # 10000
# LR = 0.00025
# BATCH_SIZE = 32
# EPSILON_MIN = 0.1
# EPSILON_MAX = 1.
# SCREEN_SIZE = (84, 84)  # (width, height)
# NO_OP_MAX = 30
# GRAYSCALE = True
# CLIP_REWARD = True
# FRAME_SKIP = 4
# TERMINAL_ON_LIFE_LOSS = True

# AGENT_HISTORY_LENGTH = 4
# CUDA = True
# DISCOUNT_FACTOR = 0.99

# # fit args
# N_FIT_EP = 25000
# EP_MAX_STEP = 1000
# REPLAY_START_SIZE = 500  # 50000
# SAVE_MODEL_EVERY = 3  # 200

# gym_env = gym.make(ENV_NAME)
# device = torch.device('cuda') if (CUDA and torch.cuda.is_available()) else torch.device('cpu')
# if not os.path.exists(OUTPUT_DIR):
#     os.makedirs(OUTPUT_DIR)


def main():
    args = TrainOptions().parse()
    device = torch.device('cuda') if (not args.no_cuda and torch.cuda.is_available()) else torch.device('cpu')
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    env = create_env(args)

    network = create_network(args, env.action_space.n, env.observation_space.shape)
    network.to(device)
    optimizer = Adam(network.parameters(), lr=args.lr)

    policy = AnnealedEpsilonGreedyPolicy(epsilon_max=args.epsilon_max,
                                         epsilon_min=args.epsilon_min, exploration_steps=args.exp_steps)
    memory = SimpleExperienceReplay(max_size=args.mem_max, batch_size=args.batch_size)
    logger = Logger()

    agent = create_agent(args, env, network, policy, memory, optimizer, logger)

    # train agent
    agent.learn(n_episodes=args.n_ep, ep_max_step=args.ep_max_step, replay_start_size=args.replay_start,
                save_every=args.freq_save_model, update_target_every=args.freq_target_update)


if __name__ == '__main__':
    main()
    sys.exit(0)
