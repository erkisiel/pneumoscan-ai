"""Data loading helpers.

Supports two data sources:
  1) MedMNIST PneumoniaMNIST .npz file (e.g. pneumoniamnist_128.npz)
  2) Kaggle "Chest X-Ray Images (Pneumonia)" folder layout:
        chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg
Labels: 0 = NORMAL, 1 = PNEUMONIA
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

CLASSES = ["NORMAL", "PNEUMONIA"]
MEAN, STD = [0.5], [0.25]


def build_transforms(img_size: int, train: bool):
    t = [transforms.Resize((img_size, img_size))]
    if train:
        t += [
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
        ]
    t += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(t)


class NpzDataset(Dataset):
    def __init__(self, npz_path, split, transform):
        d = np.load(npz_path)
        self.images = d[f"{split}_images"]          # (N, H, W) uint8
        self.labels = d[f"{split}_labels"].reshape(-1).astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = Image.fromarray(self.images[i]).convert("L")
        return self.transform(img), int(self.labels[i])


class FolderDataset(Dataset):
    def __init__(self, root, split, transform):
        self.items = []
        for label, cls in enumerate(CLASSES):
            for p in sorted((Path(root) / split / cls).glob("*")):
                if p.suffix.lower() in {".jpeg", ".jpg", ".png"}:
                    self.items.append((p, label))
        self.labels = np.array([l for _, l in self.items], dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, label = self.items[i]
        return self.transform(Image.open(p).convert("L")), label


def get_dataset(data_path, split, img_size, train=False):
    tf = build_transforms(img_size, train)
    if str(data_path).endswith(".npz"):
        return NpzDataset(data_path, split, tf)
    return FolderDataset(data_path, split, tf)


def get_loaders(data_path, img_size=128, batch_size=64, workers=2, balance=True):
    train_ds = get_dataset(data_path, "train", img_size, train=True)
    val_ds = get_dataset(data_path, "val", img_size)
    test_ds = get_dataset(data_path, "test", img_size)

    sampler = None
    if balance:  # counter class imbalance (~74% pneumonia) with balanced sampling
        counts = np.bincount(train_ds.labels, minlength=2)
        w = (1.0 / counts)[train_ds.labels]
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(w))

    kw = dict(batch_size=batch_size, num_workers=workers)
    return (
        DataLoader(train_ds, sampler=sampler, shuffle=sampler is None, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
    )
