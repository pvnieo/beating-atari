# std
from collections import deque
# 3p
import numpy as np


class SimpleExperienceReplay:
    def __init__(self, max_size, batch_size):
        self.buffer = deque(maxlen=max_size)
        self.batch_size = batch_size

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size), size=self.batch_size, replace=False)

        return [self.buffer[i] for i in index]

    def __len__(self):
        return len(self.buffer)
