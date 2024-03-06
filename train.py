from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import random
from PIL import Image

from experiment_logger import ExperimentLogger


CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")


class TripletImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, n_way_repr: int = 10, k_shot_repr: int = 10):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = list(self.root_dir.glob("*.JPG"))

        self.class_to_images = {}
        self.images_to_class = {}
        for img_path in self.image_paths:
            class_id = img_path.stem[:4]
            if class_id not in self.class_to_images:
                self.class_to_images[class_id] = []
            self.class_to_images[class_id].append(img_path)
            self.images_to_class[img_path] = class_id
        self.repr_cls = random.sample(list(self.class_to_images.keys()), n_way_repr)
        self.repr_ds = []
        for cls in self.repr_cls:
            images = random.sample(self.class_to_images[cls], min(k_shot_repr, len(self.class_to_images[cls])))
            for img in images:
                self.repr_ds.append((img, cls))

    def __len__(self):
        return len(self.image_paths)

    def yield_repr_ds(self):
        for img_path, cls in self.repr_ds:
            yield self.transform(Image.open(img_path).convert("RGB")).to(DEVICE), cls

    def __getitem__(self, idx):
        anchor_path = self.image_paths[idx]
        while True:
            positive_path = random.choice(self.class_to_images[self.images_to_class[anchor_path]])
            if positive_path != anchor_path:
                break
        negative_classes = {key: value for key, value in self.class_to_images.items() if
                            key != self.images_to_class[anchor_path]}
        negative_path = random.choice(self.class_to_images[random.choice(list(negative_classes.keys()))])
        anchor_img = Image.open(anchor_path).convert("RGB")
        positive_img = Image.open(positive_path).convert("RGB")
        negative_img = Image.open(negative_path).convert("RGB")

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img.to(DEVICE), positive_img.to(DEVICE), negative_img.to(DEVICE)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)

        # Batch normalization layers
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.batchnorm4 = nn.BatchNorm2d(128)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.relu(self.batchnorm2(self.conv2(x)))
        x = self.relu(self.batchnorm3(self.conv3(x)))
        x = self.relu(self.batchnorm4(self.conv4(x)))
        return x


if __name__ == "__main__":
    logger = ExperimentLogger()
    logger.create_experiment()

    batch_size = 32
    learning_rate = 0.001
    num_epochs = 2

    transform = transforms.Compose([
        transforms.RandomRotation(90),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(5),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    dataset = TripletImageDataset(root_dir="./data/images", transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    encoder = Encoder().to(DEVICE)
    triplet_loss = nn.TripletMarginLoss(margin=1.0)

    optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        epoch_loss = []
        tq = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", postfix={"loss": 0})
        for anchor, positive, negative in tq:
            optimizer.zero_grad()

            anchor_encoding = encoder(anchor)
            positive_encoding = encoder(positive)
            negative_encoding = encoder(negative)

            loss = triplet_loss(anchor_encoding, positive_encoding, negative_encoding)

            loss.backward()

            optimizer.step()

            epoch_loss.append(loss.item())

            tq.set_postfix({"loss": f"{round(np.average(epoch_loss), 4)}±{round(np.std(epoch_loss), 4)}"})

    print("Training complete.")
    logger.close_experiment()
