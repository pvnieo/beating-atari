# stdlib
import sys
from os import makedirs
# 3p
import torch
from torch.optim import Adam
import gym
# project
from algorithms.dqn import DQN
from envs.gym_env import GymEnv
from networks.dqn_15 import DQN15
from utils.epsilon_greedy import AnnealedEpsilonGreedyPolicy
from utils.experience_replay import SimpleExperienceReplay
from utils.logger import Logger

# ### global constant ### #
OUTPUT_DIR = "outputs"
ENV_NAME = 'Breakout-v4'

EXPLORATION_STEPS = 500
MEM_MAX_SIZE = 1000
LR = 0.001
BATCH_SIZE = 64
EPSILON_MIN = 0.1
EPSILON_MAX = 1.
SCREEN_SIZE = (84, 84)
NO_OP_MAX = 30
GRAYSCALE = True
CLIP_REWARD = True
FRAME_SKIP = 4
TERMINAL_ON_LIFE_LOSS = True

AGENT_HISTORY_LENGTH = 4
CUDA = True
DISCOUNT_FACTOR = 0.99

# fit args
N_FIT_EP = 200
EP_MAX_STEP = 1000
REPLAY_START_SIZE = 10000

gym_env = gym.make('Breakout-v4')
device = torch.device('cuda') if (CUDA and torch.cuda.is_available()) else torch.device('cpu')


def main():
    env = GymEnv(gym_env, terminal_on_life_loss=TERMINAL_ON_LIFE_LOSS, noop_max=NO_OP_MAX, frame_skip=FRAME_SKIP,
                 num_stack=AGENT_HISTORY_LENGTH, screen_size=SCREEN_SIZE, grayscale=GRAYSCALE, clip_reward=CLIP_REWARD)
    model = DQN15(SCREEN_SIZE, nc=AGENT_HISTORY_LENGTH, num_actions=env.action_space.n)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=LR)
    policy = AnnealedEpsilonGreedyPolicy(epsilon_max=EPSILON_MAX, epsilon_min=EPSILON_MIN, exploration_steps=EXPLORATION_STEPS)
    memory = SimpleExperienceReplay(max_size=MEM_MAX_SIZE, batch_size=BATCH_SIZE)
    logger = Logger()

    # create agent
    dqn_agent = DQN(env, model, policy, memory, optimizer, OUTPUT_DIR, logger, DISCOUNT_FACTOR)

    # train agent
    dqn_agent.fit(n_episodes=N_FIT_EP, ep_max_step=EP_MAX_STEP, replay_start_size=REPLAY_START_SIZE)


if __name__ == '__main__':
    main()
    sys.exit(0)
