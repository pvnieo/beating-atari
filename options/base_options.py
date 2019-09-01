# stdlib
import argparse


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Implementation of RL algorithms to beat Atari 2600 games')

    def initialize(self):
        self.parser.add_argument("--output", help="The directory where all the outputs are saved", type=str, default="outputs")
        self.parser.add_argument("--env", help="Environnement name", type=str, default="Breakout-v4")
        self.parser.add_argument("--network", help="Name of NN used to approximate Q function", type=str,
                                 choices=["simple_fc", "dqn_nips", "dqn_nature"], default="dqn_nature")
        self.parser.add_argument("--agent", help="RL Algorithm used", type=str,
                                 choices=["dqn_nips", "dqn", "double_dqn"], default="dqn")
        self.parser.add_argument("--discount_factor", help="Discount factor", type=float, default=0.99)

        return self.parser
