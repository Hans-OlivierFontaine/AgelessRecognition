import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import random
from PIL import Image


class TripletImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
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

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        anchor_path = self.image_paths[idx]
        while True:
            positive_path = random.choice(self.class_to_images[self.images_to_class[anchor_path]])
            if positive_path != anchor_path:
                break
        negative_classes = {key: value for key, value in self.class_to_images.items() if key != self.images_to_class[anchor_path]}
        negative_path = random.choice(self.class_to_images[random.choice(list(negative_classes.keys()))])
        anchor_img = Image.open(anchor_path)
        positive_img = Image.open(positive_path)
        negative_img = Image.open(negative_path)

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = TripletImageDataset(root_dir="./data/images", transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    for anchor, positive, negative in dataloader:
        pass
