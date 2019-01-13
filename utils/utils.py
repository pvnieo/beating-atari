# std
import os
# 3p


# ------------------- Update epsilon -------------------
def update_epsilon(epsilon, EXPLORATION_DECAY, EXPLORATION_MIN):
    epsilon -= EXPLORATION_DECAY
    return max(EXPLORATION_MIN, epsilon)


# --------------- Create checkpoint directory ---------------
CHECKPOINT_DIR = 'checkpoints'
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)


def create_dir_in_checkpoint(name):
    path = os.path.join(CHECKPOINT_DIR, name)
    if not os.path.exists(path):
        os.makedirs(path)


def number_dir(path):
    return len(os.listdir(path))