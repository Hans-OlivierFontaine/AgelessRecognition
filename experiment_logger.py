import sys
import shutil
from datetime import datetime
from pathlib import Path
import imageio
import torch


class ExperimentLogger:
    def __init__(self, base_dir="./experiments"):
        self.base_dir = Path(base_dir)
        self.exp_dir = None
        self.prototypes_dir = None
        self.log_file = None

    def create_experiment(self, exp_name=None):
        # Create a unique experiment directory based on timestamp
        if exp_name is None:
            exp_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.exp_dir = self.base_dir / exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.prototypes_dir = self.exp_dir / "prototypes"
        self.prototypes_dir.mkdir(parents=True, exist_ok=True)

        # Create a log file in the experiment directory
        self.log_file = self.exp_dir / "log.txt"
        sys.stdout = open(self.log_file, "w")
        sys.stderr = open(self.log_file, "a")  # Redirect stderr to the same log file

        print(f"Experiment started at: {exp_name}")

    def log_image(self, image_path, image_name):
        # Copy the image to the experiment directory
        dest_path = self.exp_dir / image_name
        shutil.copy(image_path, dest_path)
        print(f"Image saved: {dest_path}")
        Path(image_path).unlink()

    def log_weights(self, weights, filename):
        # Copy the image to the experiment directory
        dest_path = self.exp_dir / filename
        torch.save(weights, dest_path)
        print(f"Weights saved: {dest_path}")

    def log_embedding(self, embedding, cls_name, n_composition):
        # Copy the image to the experiment directory
        dest_path = self.prototypes_dir / f"{cls_name}_{n_composition}.pth"
        torch.save(embedding, dest_path)
        print(f"Prototype {cls_name} saved: {dest_path}")

    def create_video_from_images(self, duration_per_frame=0.25):
        # Get all image paths in the directory and sort them by filename
        image_paths = sorted(self.exp_dir.glob('tsne-repr_*.png'))

        # Read images and append them to a list
        print("Getting images")
        images = []
        for image_path in image_paths:
            images.append(imageio.imread(image_path))

        # Create video from images
        print("Creating video: ", (self.exp_dir / 'tsne-repr.mp4').__str__())
        with imageio.get_writer((self.exp_dir / 'tsne-repr.mp4').__str__(), fps=1 / duration_per_frame) as writer:
            for image in images:
                writer.append_data(image)

    def close_experiment(self):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        print(f"Experiment completed. Log file saved: {self.log_file}")


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
