from pathlib import Path

import torch

from experiment_logger import ExperimentLogger
from encoder import Encoder
from experiment_logger import choose_exp

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")


if __name__ == "__main__":
    exp_name = None
    exp_name = choose_exp(exp_name)
    logger = ExperimentLogger()
    logger.create_experiment(exp_name)

    encoder = Encoder().to(DEVICE)

    logger.close_experiment()
