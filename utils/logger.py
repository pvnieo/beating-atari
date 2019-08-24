# stdlib
from os.path import join, exists
from os import makedirs
from datetime import datetime
# 3p
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self):
        self.tb_writer = SummaryWriter()

    def log_dir(self, out_dir):
        self.dir = join(out_dir, "logs", str(datetime.now())[:16])
        if not exists(self.dir):
            makedirs(self.dir)
        self.tb_writer = SummaryWriter(log_dir=self.dir)

    def __del__(self):
        self.tb_writer.flush()
        self.tb_writer.close()
