# stdlib
import logging
from os.path import join, exists
from os import makedirs
from datetime import datetime
# 3p
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self):
        self.tb_writer = SummaryWriter()

    def add_agent_dir(self, out_dir):
        self.tb_dir = join(out_dir, "tb_logs", str(datetime.now())[:16])
        self.log_dir = join(out_dir, "logs")
        if not exists(self.tb_dir):
            makedirs(self.tb_dir)
        if not exists(self.log_dir):
            makedirs(self.log_dir)
        self.tb_writer = SummaryWriter(log_dir=self.tb_dir)

        # Create a custom logger
        self.logger = logging.getLogger("logger")

        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(join(self.log_dir, str(datetime.now())[:16] + ".txt"))
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.DEBUG)

        # Create formatters and add it to handlers
        c_format = logging.Formatter('%(levelname)s - %(message)s')
        f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)
        self.log('This is an info', "info")
        self.log('This is a warning', "warning")
        self.log('This is an error', "error")

    def log(self, message, level="debug"):
        if level == "debug":
            self.logger.debug(message)
        elif level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "critical":
            self.logger.critical(message)

    def __del__(self):
        self.tb_writer.flush()
        self.tb_writer.close()
