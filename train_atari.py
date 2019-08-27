# stdlib
import sys
import os
# 3p
import torch
from torch.optim import Adam
import gym
# project
from agents.dqn import DQNNips, DQN
from agents.double_dqn import DoubleDQN
from envs.gym_env import GymEnv
from networks.dqn_nature import DQNNatureNetwork
from networks.dqn_nips import DQNNipsNetwork
from utils.epsilon_greedy import AnnealedEpsilonGreedyPolicy
from utils.experience_replay import SimpleExperienceReplay
from utils.logger import Logger

# ################### Argument parser will be added later ################### #

# ### global constant ### #
OUTPUT_DIR = "outputs"
ENV_NAME = 'Breakout-v4'
network = "dqn_nature"  # [dqn_nature, dqn_nips]
agent_name = "double_dqn"  # [dqn, dqn_nips, double_dqn]

EXPLORATION_STEPS = 1000000
MEM_MAX_SIZE = 1000000
TARGET_NETWORK_UPDATE = 10  # 10000
LR = 0.00025
BATCH_SIZE = 32
EPSILON_MIN = 0.1
EPSILON_MAX = 1.
SCREEN_SIZE = (84, 84)  # (width, height)
NO_OP_MAX = 30
GRAYSCALE = True
CLIP_REWARD = True
FRAME_SKIP = 4
TERMINAL_ON_LIFE_LOSS = True

AGENT_HISTORY_LENGTH = 4
CUDA = True
DISCOUNT_FACTOR = 0.99

# fit args
N_FIT_EP = 25000
EP_MAX_STEP = 1000
REPLAY_START_SIZE = 500  # 50000
SAVE_MODEL_EVERY = 3  # 200

gym_env = gym.make(ENV_NAME)
device = torch.device('cuda') if (CUDA and torch.cuda.is_available()) else torch.device('cpu')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def main():
    env = GymEnv(gym_env, terminal_on_life_loss=TERMINAL_ON_LIFE_LOSS, noop_max=NO_OP_MAX, frame_skip=FRAME_SKIP,
                 num_stack=AGENT_HISTORY_LENGTH, screen_size=SCREEN_SIZE, grayscale=GRAYSCALE, clip_reward=CLIP_REWARD)

    # select model
    if network == "dqn_nature":
        model = DQNNatureNetwork(SCREEN_SIZE, nc=AGENT_HISTORY_LENGTH, num_actions=env.action_space.n)
    elif network == "dqn_nips":
        model = DQNNipsNetwork(SCREEN_SIZE, nc=AGENT_HISTORY_LENGTH, num_actions=env.action_space.n)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=LR)
    policy = AnnealedEpsilonGreedyPolicy(epsilon_max=EPSILON_MAX, epsilon_min=EPSILON_MIN, exploration_steps=EXPLORATION_STEPS)
    memory = SimpleExperienceReplay(max_size=MEM_MAX_SIZE, batch_size=BATCH_SIZE)
    logger = Logger()

    # create agent
    # [dqn, dqn_nips, double_dqn]
    if agent_name == "dqn":
        agent = DQN(env, model, policy, memory, optimizer, OUTPUT_DIR, logger, DISCOUNT_FACTOR)
    elif agent_name == "dqn_nips":
        agent = DQNNips(env, model, policy, memory, optimizer, OUTPUT_DIR, logger, DISCOUNT_FACTOR)
    elif agent_name == "double_dqn":
        agent = DoubleDQN(env, model, policy, memory, optimizer, OUTPUT_DIR, logger, DISCOUNT_FACTOR)

    # train agent
    agent.learn(n_episodes=N_FIT_EP, ep_max_step=EP_MAX_STEP, replay_start_size=REPLAY_START_SIZE,
                save_every=SAVE_MODEL_EVERY, update_target_every=TARGET_NETWORK_UPDATE)


if __name__ == '__main__':
    main()
    sys.exit(0)
