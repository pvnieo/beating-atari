# std
import csv
import os
import datetime
# 3p

LOGS_DIR = 'logs'
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)


class Logger:
    def __init__(self, args):
        name = args.model + args.game + '_{}.csv'.format(str(datetime.datetime.now()))
        self.csv_path = os.path.join(LOGS_DIR, name)
        header = ["episode", "step", "epsilon", "loss", "reward", "mem_size", "took", "tot_reward"]
        self.log(header)

    def log(self, row):
        with open(self.csv_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow(row)
