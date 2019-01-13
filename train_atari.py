# std
import argparse
from time import time
# 3p
import gym
import numpy as np
# project
from models.simple_dqn import SimpleDQN
from utils.utils import update_epsilon
from utils.logger import Logger


parser = argparse.ArgumentParser(
    description='An implementation of multiple approachs to automatically colorize B/W images',
    usage='python3 train.py <model> [<args>]')
parser.add_argument('-m', '--model', help='RL model to use',
                    choices=["simple_dqn"], default="simple_dqn")
parser.add_argument('--training-episodes',
                    help='Number of trainig episodes', type=int, default=50)
parser.add_argument(
    '--max-steps', help="Max possible steps in a episode", type=int, default=50000)
parser.add_argument(
    '--replay-start', help='Number of frames to populate the memory before training', default=128)
parser.add_argument(
    '--memory-size', help='Number of experiences the Memory can keep', type=int, default=1000000)
parser.add_argument(
    '--skip-frames', help='Number of skiped frames', type=int, default=4)
parser.add_argument('--game', help='The chosen game', type=str,
                    default='SpaceInvaders', choices=['SpaceInvaders', 'BreakoutDeterministic'])
parser.add_argument('--batch-size', help="Batch size", type=int, default=32)
parser.add_argument('--explo-steps',
                    help="Number of frames over wich the initial value of epsilon is linearly annealed",
                    type=int, default=850000)
parser.add_argument('--save_freq', help="Frequency of saving model", type=int, default=5)
parser.add_argument('--no_continue', help='Train model from scratch even if saved model exsits', action="store_true")

args = parser.parse_args()

# TRAINING HYPERPARAMETERS
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.1
EXPLORATION_TEST = 0.02
EXPLORATION_STEPS = args.explo_steps
EXPLORATION_DECAY = (EXPLORATION_MAX - EXPLORATION_MIN) / EXPLORATION_STEPS

# Create environnement
env = gym.make(args.game + '-v' + str(args.skip_frames))
n_actions = env.action_space.n

if __name__ == '__main__':
    begin = time()
    # Create model
    if args.model == 'simple_dqn':
        agent = SimpleDQN(n_actions, args)

    # Continue training
    if agent.is_model_saved and not args.no_continue:
        agent.load_model()
    
    # Create Logger
    logger = Logger(args)

    # Populate memory
    frame = env.reset()
    new_game = True

    print("######### Population Memory #########")
    for _ in range(args.replay_start):
        if new_game:
            state, stacked_frames = agent.preprocessor.stack_frames(frame, True)
            new_game = False
        else:
            state, stacked_frames = agent.preprocessor.stack_frames(frame, False, stacked_frames)

        # Perform a random action
        action = env.action_space.sample()
        next_frame, reward, is_terminal, _ = env.step(action)

        # Stack next state
        next_state, stacked_frames = agent.preprocessor.stack_frames(next_frame, False, stacked_frames)

        if is_terminal:
            next_state = np.zeros(state.shape, dtype='uint8')

            # Add to memory
            agent.memory.add((state, action, reward, next_state, is_terminal))

            # Restart env
            frame = env.reset
            new_game = True
        else:
            agent.memory.add((state, action, reward, next_state, is_terminal))
            frame = next_frame

    print("Memory size:", len(agent.memory))
    print("######### Memory Populated #########")

    # Main loop
    epsilon = 1.0 + EXPLORATION_DECAY
    rewards = []

    for episode in range(args.training_episodes):
        # Initialization
        episode_reward = []
        frame = env.reset()
        state, stacked_frames = agent.preprocessor.stack_frames(frame, True)
        step = 0
        mem_size = len(agent.memory)
        since = time()

        while step < args.max_steps:
            print("Episode: {} | Num steps: {}\r".format(episode, step), end="")
            step += 1
            epsilon = update_epsilon(epsilon, EXPLORATION_DECAY, EXPLORATION_MIN)

            # One q_iteration
            state, reward, is_terminal, loss = agent.q_iteration(env, state, stacked_frames, args.batch_size, epsilon)
            episode_reward.append(reward)

            # Log
            logger.log([episode, step, epsilon, loss, reward, mem_size, 0, 0])

            if is_terminal:
                took = round(time() - since, 2)
                mem_size = len(agent.memory)
                total_reward = np.sum(episode_reward)
                rewards.append(total_reward)
                print("Episode: {} | Num steps: {} | Epsilon: {} | Reward: {} | Took: {}s | Mem size: {}".format(episode, step,
                      epsilon, total_reward, took, mem_size))
                logger.log([episode, step, epsilon, loss, reward, mem_size, took, total_reward])
                break

        if (episode + 1) % args.save_freq == 0:
            agent.save_model()
    tot = int(time() - begin)
    print("Training Took:", tot)
