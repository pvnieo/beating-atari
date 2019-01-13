# std
import argparse
from time import time, sleep
# 3p
import gym
# project
from models.simple_dqn import SimpleDQN


parser = argparse.ArgumentParser(
    description='Test multiple RL algorithms to beat Atari 2600 games')
parser.add_argument('-m', '--model', help='RL model to use',
                    choices=["simple_dqn"], default="simple_dqn")
parser.add_argument('--test-episodes',
                    help='Number of test episodes', type=int, default=5)
parser.add_argument(
    '--skip-frames', help='Number of skiped frames', type=int, default=4)
parser.add_argument('--game', help='The chosen game', type=str,
                    default='SpaceInvaders', choices=['SpaceInvaders', 'BreakoutDeterministic'])
parser.add_argument('--explo-steps',
                    help="Number of frames over wich the initial value of epsilon is linearly annealed",
                    type=int, default=850000)
parser.add_argument('--no_render', help='Disable rendering game', action="store_true")

args = parser.parse_args()


# Create environnement
env = gym.make(args.game + '-v' + str(args.skip_frames))
n_actions = env.action_space.n

if __name__ == '__main__':
    begin = time()
    # Create model
    if args.model == 'simple_dqn':
        agent = SimpleDQN(n_actions, args, train=False)

    # Load model
    agent.load_model()

    for episode in range(args.test_episodes):
        # Initialization
        episode_reward = []
        frame = env.reset()

        # render env
        if not args.no_render:
            env.render()

        state, stacked_frames = agent.preprocessor.stack_frames(frame, True)
        step = 0
        since = time()

        while True:
            print("Episode: {} | Num steps: {}\r".format(episode, step), end="")
            step += 1

            # Best action
            action = agent.choose_best_action(state)
            # excute action
            next_frame, reward, is_terminal, _ = env.step(action)

            # render env
            if not args.no_render:
                env.render()
                sleep(0.1)

            episode_reward.append(reward)

            if is_terminal:
                took = round(time() - since, 2)
                total_reward = sum(episode_reward)
                print("Episode: {} | Num steps: {} | Reward: {} | Took: {}s".format(episode, step, total_reward, took))
                break

            # next state
            state, stacked_frames = agent.preprocessor.stack_frames(next_frame, False, stacked_frames)

    tot = int(time() - begin)
    print("Training Took:", tot)
