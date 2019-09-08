# 3p
import numpy as np
import torch.nn as nn


class FCNetwork(nn.Module):
    """implementation of a simple fully connected network to use it
       to debug implemented agents on simple games as cartpole
    """
    def __init__(self, input_shape, nc=4, num_actions=18):
        super().__init__()

        self.input_shape = (nc,) + tuple(input_shape)
        self.num_actions = num_actions

        self.net = nn.Sequential(
            nn.Linear(np.prod(self.input_shape), 24),
            nn.ReLU(True),
            nn.Linear(24, 24),
            nn.ReLU(True),
            nn.Linear(24, self.num_actions)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.net(x)
        return x
