# std
from os.path import join
# 3p
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, merge, multiply
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
# project
from models.base_model import BaseModel
from utils.atari_preprocessor import AtariPreprocessor
from utils.experience_replay import Memory
from utils.utils import create_dir_in_checkpoint, CHECKPOINT_DIR, number_dir


class SimpleDQN(BaseModel):
    def __init__(self, n_actions, args):
        self.preprocessor = AtariPreprocessor()
        self.n_actions = n_actions

        # Experience replay
        self.memory = Memory(args.memory_size)

        # Create check directory
        self.save_dir = self.name + '_' + args.game
        create_dir_in_checkpoint(self.save_dir)
        create_dir_in_checkpoint(self.save_dir + "/board")
        n = number_dir(join(CHECKPOINT_DIR, self.save_dir, "board"))
        self.save_board = join(CHECKPOINT_DIR, self.save_dir, "board", str(n+1))
        self.save_file = join(CHECKPOINT_DIR, self.save_dir, "model.h5")

        # Define model
        self.model = self._define_model(n_actions)
        optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        self.model.compile(optimizer, loss='mse')
        self.board = TensorBoard(log_dir=self.save_board)
        # self.ckp = ModelCheckpoint(
        #     filepath=self.best_file, verbose=1, save_best_only=True, save_weights_only=True)

    @property
    def name(self):
        return 'simple_dqn'

    def _define_model(self, n_actions):
        # Input layers
        INPUT_SHAPE = self.preprocessor.INPUT_SHAPE + (self.preprocessor.HISTORY_LENGTH,)
        input_frames = Input(INPUT_SHAPE, name='frames')
        input_actions = Input((n_actions,), name='mask')
        model = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(input_frames)
        model = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(model)
        model = Flatten()(model)
        model = Dense(256, activation='relu')(model)
        model = Dense(n_actions)(model)
        # Multiply output with the mask
        model = multiply([model, input_actions])
        model = Model(inputs=[input_frames, input_actions], outputs=model)

        return model
