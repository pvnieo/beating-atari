# std
from copy import deepcopy
from os import listdir, makedirs, remove
from os.path import join, isfile, exists
# 3p
import numpy as np
import torch
from tqdm import tqdm
# project


class BaseModel:
    def __init__(self, outputs_dir, logger):
        self.outputs_dir = join(outputs_dir, self.name)
        if not exists(self.outputs_dir):
            makedirs(self.outputs_dir)
        else:
            for file in [f for f in listdir(self.outputs_dir) if (isfile(join(self.outputs_dir, f)) and (self.name+"_") in f)]:
                remove(join(self.outputs_dir, file))
        self.logger = logger
        self.logger.add_agent_dir(self.outputs_dir)

    @property
    def name(self):
        pass

    def save(self):
        pass

    def load(self):
        pass


class DQNBasedModel(BaseModel):
    def __init__(self, env, model, policy, memory, optimizer, outputs_dir, logger, discount_factor):
        super().__init__(outputs_dir, logger)
        self.env = env
        self.online_net = model
        self.target_net = deepcopy(model)
        self.update_target_net()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False
        self.policy = policy
        self.memory = memory
        self.optimizer = optimizer
        self.discount_factor = discount_factor

    def save(self):
        torch.save(self.online_net.state_dict(), join(self.outputs_dir, f"{self.name}.pth"))
        n = len([f for f in listdir(self.outputs_dir) if (isfile(join(self.outputs_dir, f)) and self.name in f)])
        torch.save(self.online_net.state_dict(), join(self.outputs_dir, f"{self.name}_{n}.pth"))
        self.logger.log("Model saved!")

    def load(self):
        if isfile(join(self.outputs_dir, f"{self.name}.pth")):
            self.online_net.load_state_dict(torch.load(join(self.outputs_dir, f"{self.name}.pth")))
        self.logger.log("Model loaded!")

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.logger.log("Target network saved!")

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()

    def predict_action(self, state):
        return torch.argmax(self.online_net(state.unsqueeze(0)))

    def populate_memory(self, replay_start_size):
        state = self.env.reset()
        for _ in tqdm(range(replay_start_size), desc="Populating Memory: "):
            # Perform a random action
            action = self.env.action_space.sample()
            next_state, reward, is_terminal, _ = self.env.step(action)

            if is_terminal:
                self.memory.add((state, action, reward, next_state, is_terminal))
                # Restart env
                state = self.env.reset()
            else:
                self.memory.add((state, action, reward, next_state, is_terminal))
                state = next_state

    def learn(self, n_episodes, ep_max_step, replay_start_size, save_every, update_target_every):
        # populate memory
        self.populate_memory(replay_start_size)

        # main training loop
        self.train()
        self.total_steps = 0
        self.logger.log("Begin learning ...", level="info")

        for episode in range(1, n_episodes+1):
            ep_reward, ep_loss = [], []
            state = self.env.reset()
            step = 0
            while True:
                step += 1
                self.total_steps += 1
                if self.policy.explore():
                    action = self.env.action_space.sample()
                else:
                    action = self.predict_action(state)

                # excute action
                next_state, reward, is_terminal, _ = self.env.step(action)
                ep_reward.append(reward)
                self.memory.add((state, action, reward, next_state, is_terminal))

                # training part
                batch = self.memory.sample()
                states = torch.stack([x[0] for x in batch], axis=0)
                actions = [x[1] for x in batch]
                rewards = [x[2] for x in batch]
                next_states = torch.stack([x[3] for x in batch], axis=0)
                is_terminals = [x[4] for x in batch]
                loss = self.fit_batch(states, actions, rewards, next_states, is_terminals)
                ep_loss.append(loss)

                if is_terminal or (step > ep_max_step):
                    # log episode
                    self.logger.tb_writer.add_scalars("history", {"reward": sum(ep_reward), "loss": np.mean(
                        ep_loss), "steps": step, "mem_size": len(self.memory)}, episode)
                    self.logger.log(f"Episode {episode}/{n_episodes} ({int(episode/n_episodes * 100)}%) :: " +
                        f"reward: {sum(ep_reward)} | steps: {step} | total_steps: {self.total_steps}", level="info")

                    # save models
                    if self.total_steps % save_every == 0:
                        self.save()
                    if self.total_steps % update_target_every == 0:
                        self.update_target_net()

                    break

        self.save()

    def fit_batch(self, states, actions, rewards, next_states, is_terminals):
        return 0
