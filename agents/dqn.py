# 3p
import torch
from torch.nn.functional import smooth_l1_loss
# project
from .base_model import DQNBasedModel


class DQNNips(DQNBasedModel):
    def __init__(self, env, network, policy, memory, optimizer, outputs_dir, logger, discount_factor=0.99):
        super().__init__(env, network, policy, memory, optimizer, outputs_dir, logger, discount_factor)

    @property
    def name(self):
        return "dqn_nips"

    def fit_batch(self, states, actions, rewards, next_states, is_terminals):
        target_q_values = self.online_net(next_states)
        # If terminal, we use y_i = r_i instead of y_i = r_i + gamma * max Q
        target_q_values[is_terminals] = 0
        # Compute targets: y_i = r_i + gamma * max Q
        target_q_values = (torch.FloatTensor(rewards) +
                           self.discount_factor * torch.max(target_q_values, dim=1)[0]).reshape(-1, 1)
        # compute loss
        predicted_q_values = torch.gather(self.online_net(states), 1, torch.LongTensor(actions).reshape(-1, 1))
        loss = smooth_l1_loss(predicted_q_values, target_q_values)
        # optimize
        self.optimizer.zero_grad()
        loss.backward()

        return loss.item()

    def update_target_net(self):
        pass


class DQN(DQNBasedModel):
    def __init__(self, env, network, policy, memory, optimizer, outputs_dir, logger, discount_factor=0.99):
        super().__init__(env, network, policy, memory, optimizer, outputs_dir, logger, discount_factor)

    @property
    def name(self):
        return "dqn"

    def fit_batch(self, states, actions, rewards, next_states, is_terminals):
        target_q_values = self.target_net(next_states)
        # If terminal, we use y_i = r_i instead of y_i = r_i + gamma * max Q
        target_q_values[is_terminals] = 0
        # Compute targets: y_i = r_i + gamma * max Q-
        target_q_values = (torch.FloatTensor(rewards) +
                           self.discount_factor * torch.max(target_q_values, dim=1)[0]).reshape(-1, 1)
        # compute loss
        predicted_q_values = torch.gather(self.online_net(states), 1, torch.LongTensor(actions).reshape(-1, 1))
        loss = smooth_l1_loss(predicted_q_values, target_q_values)
        # optimize
        self.optimizer.zero_grad()
        loss.backward()

        return loss.item()
