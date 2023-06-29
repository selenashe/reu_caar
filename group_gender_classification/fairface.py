import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch

class Fairface(Dataset):
    def __init__(self, img_dir, annotations_file, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(annotations_file)
        self.transform = transform
        self.age_map = {
            '0-2': 0, '3-9': 1, '10-19': 2,
            '20-29': 3, '30-39': 4, '40-49': 5,
            '50-59': 6, '60-69': 7, 'more than 70': 8
        }
        self.race_map = {
            'White': 0,
            'Southeast Asian': 1,
            'Black': 2,
            'Middle Eastern': 3,
            'Indian': 4,
            'Latino_Hispanic': 5,
            'East Asian': 6,
        }

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path).float() / 255.0
        age_idx = self.age_map[self.img_labels.iloc[idx, 1]]
        gender = 1 if self.img_labels.iloc[idx, 2] == 'Male' else 0
        race_idx = self.race_map[self.img_labels.iloc[idx, 3]]
        label = torch.tensor([age_idx, gender, race_idx])
        if self.transform:
            image = self.transform(image)
        return image, label
