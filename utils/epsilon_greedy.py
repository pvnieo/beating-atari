# stdlib
from random import random


class AnnealedEpsilonGreedyPolicy:
    def __init__(self, epsilon_max=1., epsilon_min=0.1, exploration_steps=50000, decay_type="linear"):
        self.epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.decay_type = decay_type
        if self.decay_type == "linear":
            self.exploration_decay = (epsilon_max - epsilon_min) / exploration_steps
        elif self.decay_type == "exp":
            self.exploration_decay = (epsilon_min / epsilon_max) ** (1 / exploration_steps)

    def step(self):
        if self.epsilon == self.epsilon_min:
            return
        if self.decay_type == "linear":
            self.epsilon -= self.exploration_decay
        elif self.decay_type == "exp":
            self.epsilon *= self.exploration_decay

    def explore(self):
        self.step()
        return random() < self.epsilon
