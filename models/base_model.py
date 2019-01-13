# std
from os.path import join, isfile
# 3p
import numpy as np
from keras.utils import to_categorical
# project


class BaseModel:
    def __init__(self):
        self.model = None
        self.memory = None
        self.preprocessor = None
        self.save_dir = None

    @property
    def name(self):
        pass

    def fit_batch(self, states, actions, rewards, next_states, is_terminals, gamma=0.99):
        """Do one iteration of Q-learning algorithm

        Inputs:
        -------
        states:
            numpy array of starting states
        actions:
            numpy array of one-hot encoded actions corresponding to the start states
        rewards:
            numpy array of rewards corresponding to the start states and actions
        is_terminal:
            numpy boolean array of whether the resulting state is terminal
        next_states:
            numpy array of the resulting states corresponding to the start states and actions
        """
        # Compute target Q values
        target_q_values = self.model.predict(
            [next_states, np.ones(actions.shape)])
        # If terminal, we use y_i = r_i instead of y_i = r_i + gamma * max Q
        target_q_values[is_terminals.astype(np.int)] = 0
        # Compute targets: y_i = r_i + gamma * max Q
        target = rewards + gamma * np.max(target_q_values, axis=1)
        # Fit model
        history = self.model.fit(
            [states, actions], actions * target[:, None],
            epochs=1, batch_size=len(states), verbose=0,
            callbacks=[self.board]
        )
        return history.history['loss'][0]

    def q_iteration(self, env, state, stacked_frames, batch_size, epsilon):
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = self.choose_best_action(state)

        # excute action
        next_frame, reward, is_terminal, _ = env.step(action)

        if is_terminal:
            next_frame = np.zeros(next_frame.shape, dtype='uint8')

        # Remember experience
        next_state, stacked_frames = self.preprocessor.stack_frames(
            next_frame, False, stacked_frames)
        self.memory.add((state, action, reward, next_state, is_terminal))

        # Learning part
        batch = self.memory.sample(batch_size)
        states = np.stack(batch[:, 0], axis=0)
        actions = to_categorical(batch[:, 1], num_classes=self.n_actions)
        rewards = batch[:, 2]
        next_states = np.stack(batch[:, 3], axis=0)
        is_terminals = batch[:, 4]

        # Preprocess states
        states = self.preprocessor.preprocess_batch(states)
        next_states = self.preprocessor.preprocess_batch(next_states)

        loss = self.fit_batch(states, actions, rewards, next_states, is_terminals)

        return next_state, reward, is_terminal, loss

    def choose_best_action(self, state):
        state = np.expand_dims(state, axis=0)
        actions = np.expand_dims(np.ones(self.n_actions), axis=0)
        predicted_q = self.model.predict([state, actions])
        action = np.argmax(predicted_q)
        return action

    def predict(self):
        pass

    def save_model(self):
        self.model.save_weights(self.save_file)
        print("Model saved!")

    def is_model_saved(self):
        return isfile(self.save_file)

    def load_model(self):
        print("Loading model")
        self.model.load_weights(self.save_file)
