# project
from .dqn import DQN, DQNNips
from .double_dqn import DoubleDQN


def create_agent(args, env, network, policy, memory, optimizer, logger,):
    if args.agent == "dqn_nips":
        agent = DQNNips(env, network, policy, memory, optimizer, args.output, logger, args.discount_factor)
    elif args.agent == "dqn":
        agent = DQN(env, network, policy, memory, optimizer, args.output, logger, args.discount_factor)
    elif args.agent == "double_dqn":
        agent = DoubleDQN(env, network, policy, memory, optimizer, args.output, logger, args.discount_factor)

    return agent
