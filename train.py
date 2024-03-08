from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from experiment_logger import ExperimentLogger
from encoder import Encoder
from dataset import FlatDataset, TripletImageDataset
from visualization import tsne_exec
from prorotype_utils import compute_prototypes, compute_own_class_prototype_probabilities
from stats import print_probability_statistics


CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")


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
            tsne_exec(embeddings, labels, epoch, logger)

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
        prototypes = compute_prototypes(embeddings, labels, logger)
        probs = compute_own_class_prototype_probabilities(prototypes, embeddings, labels)
        print_probability_statistics(probs, 0.5)
    print("Evaluation completed")
    logger.close_experiment()
