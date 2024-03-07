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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from experiment_logger import ExperimentLogger


CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")


class FlatDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = list(self.root_dir.glob("*.JPG"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img.to(DEVICE), Path(img_path).name[:4]


class TripletImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, no_augment=None, n_way_repr: int = 10, k_shot_repr: int = 10):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.no_augment = no_augment
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
            yield torch.unsqueeze(self.no_augment(Image.open(img_path).convert("RGB")).to(DEVICE), 0), [cls]

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

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.relu(self.batchnorm2(self.conv2(x)))
        x = self.relu(self.batchnorm3(self.conv3(x)))
        x = self.relu(self.batchnorm4(self.conv4(x)))
        x = self.flatten(x)
        # x = x.view(x.size(0), -1)
        return x


def tsne_exec(m_embeddings, m_labels, m_epoch):
    tsne = TSNE(n_components=2, random_state=42)
    # Reduce dimensionality using t-SNE
    m_embeddings = np.squeeze(m_embeddings, axis=1)
    m_labels = np.squeeze(m_labels, axis=1)
    embeddings_2d = tsne.fit_transform(m_embeddings)

    # Plot 2D embeddings
    plt.figure(figsize=(10, 8))
    for label in np.unique(m_labels):
        embeddings_to_plot = embeddings_2d[m_labels == label]
        plt.scatter(embeddings_to_plot[:, 0], embeddings_to_plot[:, 1], label=label)
    plt.title("t-SNE Visualization of Embeddings")
    plt.legend()
    plt.savefig("./tmp.png")
    logger.log_image("./tmp.png", f"tsne-repr_{m_epoch}.png")


def compute_prototypes(m_embeddings, m_labels):
    unique_labels = np.unique(m_labels)
    m_prototypes = {}
    for label in unique_labels:
        indices = np.where(m_labels == label)[0]
        class_embeddings = m_embeddings[indices]
        m_prototypes[label] = np.mean(class_embeddings, axis=0)
    return m_prototypes


def compute_own_class_prototype_probabilities(m_prototypes, m_embeddings, m_labels):
    own_class_prototype_probs = []
    for i in range(len(m_embeddings)):
        embedding = m_embeddings[i]
        label = m_labels[i]
        true_prototype = m_prototypes[str(label[0])]
        true_dst = np.linalg.norm(embedding - true_prototype)
        false_dst = np.inf
        for cls_label in m_prototypes.keys():
            if cls_label == str(label[0]):
                continue
            query_prototype = m_prototypes[cls_label]
            false_dst = min([np.linalg.norm(embedding - query_prototype), false_dst])
        own_class_prototype_prob = false_dst / (true_dst + false_dst)
        own_class_prototype_probs.append(own_class_prototype_prob)
    return own_class_prototype_probs


def print_probability_statistics(own_probs, threshold: float = 0.5):
    min_prob = min(own_probs)
    max_prob = max(own_probs)
    mean_prob = np.mean(own_probs)
    std_prob = np.std(own_probs)

    print("Minimum probability:", min_prob)
    print("Maximum probability:", max_prob)
    print("Mean probability:", mean_prob)
    print("Standard deviation of probabilities:", std_prob)

    # Additional relevant metrics
    num_samples = len(own_probs)
    num_high_prob = sum(prob > threshold for prob in own_probs)
    num_low_prob = sum(prob < threshold for prob in own_probs)
    ratio_high_prob = num_high_prob / num_samples

    print(f"Number of samples with probability < {threshold}:", num_low_prob)
    print(f"Number of samples with probability > {threshold}:", num_high_prob)
    print(f"Ratio of samples with probability > {threshold}:", ratio_high_prob)


if __name__ == "__main__":
    logger = ExperimentLogger()
    logger.create_experiment()

    batch_size = 32
    learning_rate = 0.001
    num_epochs = 1
    represent = True
    imgsz = 256

    transform = transforms.Compose([
        transforms.RandomRotation(90),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(5),
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
    ])
    no_augment = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
    ])
    dataset = TripletImageDataset(root_dir="./data/images", transform=transform, no_augment=no_augment)
    f_dataset = FlatDataset(root_dir="./data/images", transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    f_dataloader = DataLoader(f_dataset, batch_size=1, shuffle=False)

    encoder = Encoder().to(DEVICE)
    triplet_loss = nn.TripletMarginLoss(margin=1.0)

    optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        epoch_loss = []
        tq = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", postfix={"loss": 0})
        for anchor, positive, negative in tq:
            encoder.train()
            optimizer.zero_grad()

            anchor_encoding = encoder(anchor)
            positive_encoding = encoder(positive)
            negative_encoding = encoder(negative)

            loss = triplet_loss(anchor_encoding, positive_encoding, negative_encoding)

            loss.backward()

            optimizer.step()

            epoch_loss.append(loss.item())

            tq.set_postfix({"loss": f"{round(np.average(epoch_loss), 4)}±{round(np.std(epoch_loss), 4)}"})

        if represent:
            encoder.eval()
            embeddings = []
            labels = []
            with torch.no_grad():
                for img, cls in dataset.yield_repr_ds():
                    embedding = encoder(img)
                    embeddings.append(embedding.cpu().detach().numpy())
                    labels.append(cls)
            embeddings = np.array(embeddings)
            labels = np.array(labels)
            tsne_exec(embeddings, labels, epoch)

    logger.log_weights(encoder.state_dict(), f"imgsz{imgsz}_epochs{num_epochs}.pth")

    print("Training complete.")
    logger.create_video_from_images()
    with torch.no_grad():
        encoder.eval()
        embeddings = []
        labels = []
        tq = tqdm(f_dataloader, desc=f"Evaluating comprehension", postfix={"loss": 0})
        for img, cls in tq:
            embedding = encoder(img)
            embeddings.append(embedding.cpu().detach().numpy())
            labels.append(cls)
        embeddings = np.array(embeddings)
        labels = np.array(labels)
        prototypes = compute_prototypes(embeddings, labels)
        probs = compute_own_class_prototype_probabilities(prototypes, embeddings, labels)
        print_probability_statistics(probs, 0.5)
    print("Evaluation completed")
    logger.close_experiment()
