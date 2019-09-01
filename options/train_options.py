# stdlib
import argparse
# project
from base_options import BaseOptions


class TrainOptions(BaseOptions):
    def __init__(self):
        super().__init__()

    def initialize(self):
        parser = BaseOptions.initialize()
        parser.add_argument("--exp_steps", help="Number of steps over which epsilon is linearly annealed", type=int, default=1000000)
        parser.add_argument("--epsilon_min", help="Final value of epsilon in epsilon greedy expoloration", type=float, default=0.1)
        parser.add_argument("--epsilon_max", help="First value of epsilon in epsilon greedy expoloration", type=float, default=1.)

        parser.add_argument("--batch_size", help="Minibatch size", type=int, default=32)
        parser.add_argument("--lr", help="Learning rate", type=float, default=2.5e-4)
        parser.add_argument("--freq_target_update", help="Frequency (Num of steps) at which the target network is updated", type=int, default=10000)

        parser.add_argument("--mem_max", help="Maximum size of replay memory", type=int, default=1000000)
        parser.add_argument("--no_clip_reward", help="Either to clip reward in be in {-1, 0, 1} or not", action="store_true")
        parser.add_argument("--history_length", help="Number of recent experienced frames given as input to Q network", type=int, default=4)

        parser.add_argument("--no_gray", help="Don't convert frame to grayscale", action="store_true")
        parser.add_argument("--frame_skip", help="Number of frames skipped, last action is repeated", type=int, default=4)
        parser.add_argument("--screen_height", help="The height at which the frame is resized", type=int, default=84)
        parser.add_argument("--screen_width", help="The width at which the frame is resized", type=int, default=84)
        parser.add_argument("--no_terminal_loss", help="Don't terminate episode on life loss", action="store_true")
        parser.add_argument("--no_op_max", help="Maximum number of random actions taken by the agent at the beginning of episode", type=int, default=10)
        parser.add_argument("--no_cuda", help="Deactivate GPU", action="store_true")

        parser.add_argument("--n_ep", help="Number of epsiodes to train on", type=int, default=25000)
        parser.add_argument("--ep_max_step", help="Maximum number of steps per episode", type=int, default=1000)
        parser.add_argument("--replay_start", help="Initial number of experiences to populate the replay memory", type=int, default=50000)
        parser.add_argument("--freq_save_model", help="Frequency of saving model weights (Num of steps)", type=int, default=200)
