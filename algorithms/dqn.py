# 3p
import torch
from torch.nn.functional import smooth_l1_loss
# project
from .base_model import DQNBasedModel


class DQN(DQNBasedModel):
    def __init__(self, env, model, policy, memory, optimizer, outputs_dir, logger, discount_factor=0.99):
        super().__init__(env, model, policy, memory, optimizer, outputs_dir, logger, discount_factor)

    @property
    def name(self):
        return "dqn"

    def fit_batch(self, frames, actions, rewards, next_frames, is_terminals):
        target_q_values = self.model(next_frames)
        # If terminal, we use y_i = r_i instead of y_i = r_i + gamma * max Q
        target_q_values[is_terminals] = 0
        # Compute targets: y_i = r_i + gamma * max Q
        target_q_values = torch.FloatTensor(rewards) + self.discount_factor * torch.max(target_q_values, dim=1)[0]
        # compute loss
        predicted_q_values = torch.max(self.model(frames), dim=1)[0]
        loss = smooth_l1_loss(predicted_q_values, target_q_values)
        # optimize
        self.optimizer.zero_grad()
        loss.backward()

        return loss.item()
