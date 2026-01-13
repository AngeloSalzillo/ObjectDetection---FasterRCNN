import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, image_dir, bbox_csv, classes_csv, transform=None):
        self.image_dir = image_dir
        self.image_files = os.listdir(self.image_dir)
        self.bbox_csv = bbox_csv.groupby('ImageID')
        self.classes_csv = {row[0]: i + 1 for i, row in classes_csv.iterrows()}
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_files[index])
        img = Image.open(image_path).convert("RGB")
        img_id = os.path.splitext(self.image_files[index])[0]

        if img_id not in self.bbox_csv.groups:
            return self.__getitem__((index + 1) % len(self))

        boxes, labels = [], []
        for _, row in self.bbox_csv.get_group(img_id).iterrows():
            labels.append(self.classes_csv[row["LabelName"]])

            w, h = img.size
            xmin = row["XMin"] * w
            xmax = row["XMax"] * w
            ymin = row["YMin"] * h
            ymax = row["YMax"] * h
            boxes.append([xmin, ymin, xmax, ymax])

        if len(boxes) == 0:
            return self.__getitem__((index + 1) % len(self))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

        if self.transform:
            img = self.transform(img)

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))
