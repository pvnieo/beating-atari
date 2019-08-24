# std
import os
from os.path import join
# 3p
import numpy as np
import torch
from tqdm import tqdm
# project


class BaseModel:
    def __init__(self, outputs_dir, logger):
        self.outputs_dir = join(outputs_dir, self.name)
        if not os.path.exists(self.outputs_dir):
            os.makedirs(self.outputs_dir)
        self.logger = logger
        self.logger.log_dir(self.outputs_dir)

    @property
    def name(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass


class DQNBasedModel(BaseModel):
    def __init__(self, env, model, policy, memory, optimizer, outputs_dir, logger, discount_factor):
        super().__init__(outputs_dir, logger)
        self.env = env
        self.model = model
        self.policy = policy
        self.memory = memory
        self.optimizer = optimizer
        self.discount_factor = discount_factor

    def save_model(self):
        torch.save(self.model.state_dict(), join(self.outputs_dir, f"{self.name}.pth"))
        print("Model saved!")

    def load_model(self):
        if os.path.isfile(join(self.outputs_dir, f"{self.name}.pth")):
            self.model.load_state_dict(torch.load(join(self.outputs_dir, f"{self.name}.pth")))
        print("Model loaded!")

    def predict_action(self, frame):
        return torch.argmax(self.model(frame.unsqueeze(0)))

    def populate_memory(self, replay_start_size):
        frame = self.env.reset()
        for _ in tqdm(range(replay_start_size), desc="Populating Memory: "):
            # Perform a random action
            action = self.env.action_space.sample()
            next_frame, reward, is_terminal, _ = self.env.step(action)

            if is_terminal:
                self.memory.add((frame, action, reward, next_frame, is_terminal))
                # Restart env
                frame = self.env.reset()
            else:
                self.memory.add((frame, action, reward, next_frame, is_terminal))
                frame = next_frame

    def fit(self, n_episodes, ep_max_step, replay_start_size):
        # populate memory
        self.populate_memory(replay_start_size)

        # main training loop
        self.model.train()
        for episode in tqdm(range(n_episodes), desc="Training agent: "):
            ep_reward, ep_loss = [], []
            frame = self.env.reset()
            step = 0
            while True:
                step += 1
                if self.policy.explore():
                    action = self.env.action_space.sample()
                else:
                    action = self.predict_action(frame)

                # excute action
                next_frame, reward, is_terminal, _ = self.env.step(action)
                ep_reward.append(reward)
                self.memory.add((frame, action, reward, next_frame, is_terminal))

                # training part
                batch = self.memory.sample()
                frames = torch.stack([x[0] for x in batch], axis=0)
                actions = [x[1] for x in batch]
                rewards = [x[2] for x in batch]
                next_frames = torch.stack([x[3] for x in batch], axis=0)
                is_terminals = [x[4] for x in batch]
                loss = self.fit_batch(frames, actions, rewards, next_frames, is_terminals)
                ep_loss.append(loss)

                if is_terminal or (step > ep_max_step):
                    self.logger.tb_writer.add_scalars("history", {"reward": sum(
                        ep_reward), "loss": np.mean(ep_loss), "steps": step}, episode)
                    break

        self.save_model()

    def fit_batch(self, frames, actions, rewards, next_frames, is_terminals):
        return 0
