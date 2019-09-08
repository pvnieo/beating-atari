# project
from .simple_fc import FCNetwork
from .dqn_nips import DQNNipsNetwork
from .dqn_nature import DQNNatureNetwork


def create_network(args, num_actions, input_shape):
    if args.network == "simple_fc":
        network = FCNetwork(input_shape, num_actions=num_actions, nc=args.history_length)
    elif args.network == "dqn_nips":
        network = DQNNipsNetwork((args.screen_width, args.screen_height), nc=args.history_length, num_actions=num_actions)
    elif args.network == "dqn_nature":
        network = DQNNatureNetwork((args.screen_width, args.screen_height), nc=args.history_length, num_actions=num_actions)

    return network
