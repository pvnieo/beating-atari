# 3p
import torch
import torch.nn as nn
from torch.autograd import Variable


class DQN13(nn.Module):
    """implementation of the network used in the DQN 2013 paper
    """
    def __init__(self, input_shape, nc=4, num_actions=18):
        super().__init__()

        self.input_shape = (nc,) + tuple(input_shape)
        self.num_actions = num_actions

        self.cnns = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 16, 8, stride=4),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ReLU(True),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions)
        )

    def forward(self, x):
        x = self.cnns(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        return self.cnns(Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
