import torch
from torch.utils.data import Dataset
from pathlib import Path
import random
from PIL import Image

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
