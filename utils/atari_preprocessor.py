# std
from collections import deque
# 3p
import numpy as np
import cv2


class AtariPreprocessor:
    def __init__(self):
        self.INPUT_SHAPE = (110, 84)
        self.HISTORY_LENGTH = 4

    def _preporcess_frame(self, frame):
        """Performs the preprocessing mentioned in `Human-level control through deep reinforcement learning`.

        Inputs:
        -------
        frame: numpy array, shape=(210, 160, 3)
            RGB frame from the simulator

        outputs:
        --------
        model_input: numpy array, shape=(110,84)
            Gray scale croped version of frame
        """
        # using cv2 for effeciency
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = self.INPUT_SHAPE[::-1]
        model_input = cv2.resize(gray_frame, dsize=size, interpolation=cv2.INTER_CUBIC)
        return model_input.astype('uint8')

    def preprocess_batch(self, batch):
        batch = batch.astype('float32') / 255.
        return batch

    def clip_reward(self, reward):
        return np.clip(reward, -1., 1.)

    def stack_frames(self, new_frame, is_new_episode, stacked_frames=None):
        new_frame = self._preporcess_frame(new_frame)
        length = self.HISTORY_LENGTH

        if is_new_episode:
            stacked_frames = deque([np.zeros(self.INPUT_SHAPE, dtype=np.int) for i in range(length)], maxlen=length)
            for _ in range(length):
                stacked_frames.append(new_frame)
        else:
            stacked_frames.append(new_frame)

        stacked_state = np.stack(stacked_frames, axis=2)
        return stacked_state, stacked_frames
