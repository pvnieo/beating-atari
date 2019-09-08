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
        self.env = env
        self.one_dim_state = (len(self.env.observation_space.shape) == 1)
        if self.one_dim_state:
            self.env = wrapper(env, terminal_on_life_loss=False, noop_max=0, frame_skip=0,
                               num_stack=num_stack, screen_size=None, grayscale=False, clip_reward=False)
        else:
            self.env = wrapper(env, terminal_on_life_loss=terminal_on_life_loss, noop_max=noop_max, frame_skip=frame_skip,
                               num_stack=num_stack, screen_size=screen_size, grayscale=grayscale, clip_reward=clip_reward)
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self):
        self.env.close()
        state = self.env.reset()
        if self.one_dim_state:
            state = torch.from_numpy(np.array(state)).float()
        else:
            state = torch.from_numpy(np.array(state)[:, :, :, 0]).float()
        return state

    def step(self, action):
        next_state, reward, is_terminal, _ = self.env.step(action)
        if self.one_dim_state:
            next_state = torch.from_numpy(np.array(next_state)).float()
        else:
            next_state = torch.from_numpy(np.array(next_state)[:, :, :, 0]).float()
        return next_state, reward, is_terminal, _

    def render(self,  mode='human'):
        self.env.render(mode=mode)

    def __del__(self):
        self.env.close()
