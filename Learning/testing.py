import torch
import pandas as pd
from PIL import Image
from torchvision import transforms

from models.faster_rcnn import build_model
from utils.checkpoint import load_checkpoint
from utils.visualization import plot_detections


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BASE_DIR = "/content/drive/MyDrive/Open Images Object Detection"
    CLASSES_FILE = f"{BASE_DIR}/class-descriptions-boxable.csv"
    CHECKPOINT = "fasterrcnn_checkpoint.pth"
    TEST_IMAGE = f"{BASE_DIR}/test1.jpg"

    classes_csv = pd.read_csv(CLASSES_FILE, header=None)
    idx_to_class = {i + 1: row[1] for i, row in classes_csv.iterrows()}

    model = build_model(len(classes_csv) + 1, pretrained=False, device=device)
    load_checkpoint(CHECKPOINT, model, device=device)
    model.eval()

    image = Image.open(TEST_IMAGE).convert("RGB")
    tensor = transforms.ToTensor()(image).to(device)

    with torch.no_grad():
        output = model([tensor])[0]

    keep = output["scores"] > 0.7
    boxes = output["boxes"][keep].cpu().numpy()
    labels = output["labels"][keep].cpu().numpy()
    scores = output["scores"][keep].cpu().numpy()

    plot_detections(image, boxes, labels, scores, idx_to_class)


if __name__ == "__main__":
    main()
