from pathlib import Path

import torch

from experiment_logger import ExperimentLogger


def compute_new_prototype(embedding1, weight1, embedding2, weight2: int = 1):
    weighted_embedding = (embedding1 * weight1 + embedding2 * weight2) / (weight1 + weight2)
    return weighted_embedding


def find_most_recent_experiment(experiments_dir=Path('./experiments')):
    # List all directories in the experiments directory
    experiment_dirs = [entry for entry in experiments_dir.iterdir() if entry.is_dir()]

    # Sort the directories by creation time (modification time)
    sorted_experiment_dirs = sorted(experiment_dirs, key=lambda d: d.stat().st_mtime, reverse=True)

    # Return the path to the most recent experiment directory
    assert len(sorted_experiment_dirs) > 0
    return sorted_experiment_dirs[0]


def choose_exp(m_exp_name=None):
    if m_exp_name is None:
        m_exp_name = find_most_recent_experiment()
    else:
        m_exp_name = Path(f"./experiments/{m_exp_name}")
    assert Path(f"./experiments/{m_exp_name}").exists()
    return m_exp_name


if __name__ == "__main__":
    exp_name = None
    exp_name = choose_exp(exp_name)
    logger = ExperimentLogger()
    logger.create_experiment(exp_name)

    logger.close_experiment()
