import sys
import shutil
from datetime import datetime
from pathlib import Path


class ExperimentLogger:
    def __init__(self, base_dir="./experiments"):
        self.base_dir = Path(base_dir)
        self.exp_dir = None
        self.log_file = None

    def create_experiment(self):
        # Create a unique experiment directory based on timestamp
        exp_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.exp_dir = self.base_dir / exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

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

    def close_experiment(self):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        print(f"Experiment completed. Log file saved: {self.log_file}")
