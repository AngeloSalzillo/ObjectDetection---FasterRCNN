import os
import torch
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from datasets.open_images import ImageDataset, collate_fn
from models.faster_rcnn import build_model
from utils.checkpoint import save_checkpoint


def main():
    # -------- Paths --------
    DATA_ROOT = "/content/data"  # change for local usage
    BASE_DIR = "/content/drive/MyDrive/Open Images Object Detection"

    TRAIN_DIR = os.path.join(DATA_ROOT, "train_0")
    VAL_DIR = os.path.join(DATA_ROOT, "validation")

    CLASSES_FILE = os.path.join(BASE_DIR, "class-descriptions-boxable.csv")
    TRAIN_BBOX = os.path.join(BASE_DIR, "train-annotations-bbox.csv")
    VAL_BBOX = os.path.join(BASE_DIR, "validation-annotations-bbox.csv")

    # -------- Device --------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- Data --------
    classes_csv = pd.read_csv(CLASSES_FILE, header=None)
    train_bbox = pd.read_csv(TRAIN_BBOX)
    val_bbox = pd.read_csv(VAL_BBOX)

    transform = transforms.ToTensor()

    train_dataset = ImageDataset(TRAIN_DIR, train_bbox, classes_csv, transform)
    val_dataset = ImageDataset(VAL_DIR, val_bbox, classes_csv, transform)

    num_workers = min(os.cpu_count() - 1, 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    # -------- Model --------
    num_classes = len(classes_csv) + 1
    model = build_model(num_classes, pretrained=False, device=device)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # -------- Training --------
    model.train()
    for epoch in range(1):
        for images, targets in tqdm(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch {epoch} loss: {losses.item()}")

    # -------- Validation --------
    model.eval()
    metric = MeanAveragePrecision()

    with torch.no_grad():
        for images, targets in tqdm(val_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            preds = model(images)
            metric.update(preds, targets)

    results = metric.compute()
    print("Validation mAP:", results["map"].item())

    # -------- Save --------
    save_checkpoint(
        "fasterrcnn_checkpoint.pth",
        model,
        optimizer,
        epoch,
        metrics=results,
    )


if __name__ == "__main__":
    main()
