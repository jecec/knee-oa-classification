import numpy as np
import cv2
from torch.utils.data import Dataset
import torch
from torchvision.transforms import v2

from args import get_args

args = get_args()


def read_xray(path):
    """Read xray images and resize non-uniform images with padding.
    After resizing these, resize all images to 224x224 as required by ResNet

    returns:
        xray_3ch: Greyscale xray images with suitable dimensions for ResNet.
    """
    xray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    xray_3ch = np.stack([xray, xray, xray], axis=-1)

    return xray_3ch

class Xray_dataset(Dataset):
    """Class for initializing dataset."""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path = self.dataset["Directory"].iloc[idx]
        img = read_xray(path)
        # Apply transformations if provided
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img)

        label = self.dataset["KL"].iloc[idx]
        label = torch.tensor(label, dtype=torch.long)

        return {
            'img': img,
            'label': label
        }

def transforms(phase):
    """Function for applying transformations to dataset.

    args:
        phase: Either 'train' or 'val' for respective phases.
                Only resizing and normalization are applied during validation and evaluation.
    """
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomRotation(15),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])

    ])
    val_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
    ])
    if phase == 'train':
        return train_transform
    elif phase == 'val':
        return val_transform