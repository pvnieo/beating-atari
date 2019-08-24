# stdlib
from collections import deque
# 3p
import numpy as np
import torch
# project
from .wrappers import wrapper


class Env:
    def __init__(self):
        self.action_space = None

    def reset(self):
        pass

    def step(self):
        pass


class GymEnv(Env):
    def __init__(self, env, terminal_on_life_loss=True, noop_max=30, frame_skip=4,
                 num_stack=4, screen_size=(84, 84), grayscale=True, clip_reward=True):
        self.env = wrapper(env, terminal_on_life_loss=terminal_on_life_loss, noop_max=noop_max, frame_skip=frame_skip,
                           num_stack=num_stack, screen_size=screen_size, grayscale=grayscale, clip_reward=clip_reward)
        self.action_space = env.action_space

    def reset(self):
        frame = self.env.reset()
        frame = torch.from_numpy(np.array(frame)[:, :, :, 0]).float()
        return frame

    def step(self, action):
        next_frame, reward, is_terminal, _ = self.env.step(action)
        next_frame = torch.from_numpy(np.array(next_frame)[:, :, :, 0]).float()
        return next_frame, reward, is_terminal, _

    def __del__(self):
        self.env.close()
