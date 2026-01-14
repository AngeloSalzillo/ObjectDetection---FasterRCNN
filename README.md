# **Faster R-CNN Object Detection on Open Images**

This repository contains a PyTorch implementation of Faster R-CNN trained and evaluated on the Open Images Dataset.
The project was originally developed in Google Colab and later refactored into a modular, reusable, and GitHub-friendly structure.

The code supports:
- Training Faster R-CNN on Open Images bounding box annotations
- Validation using mAP (Mean Average Precision)
- Inference and visualization on custom images
- Checkpoint saving and loading


## **Requirements**

Create a Python environment (Python ≥ 3.9) and install dependencies.
Main dependencies:
- torch
- torchvision
- torchmetrics
- pandas
- Pillow
- matplotlib
- tqdm


## **Dataset Preparation**

This project uses the Open Images Dataset (bounding box annotations and image folders).

Required files:
- train-annotations-bbox.csv
- validation-annotations-bbox.csv
- class-descriptions-boxable.csv

Extracted image folders:
- train_x/   (with x = 1,..,f)
- validation/

Since the entire train dataset provided by Open Images is huge, it was split in several chunks (from train_0 to train_f).



## **Training**

Run training and validation:
python training/train.py


What happens:
- Loads Open Images annotations
- Trains Faster R-CNN (ResNet-50 FPN backbone)
- Evaluates on validation set using mAP
- Saves a checkpoint:
    fasterrcnn_checkpoint.pth


Checkpoint includes:
- Model weights
- Optimizer state
- Epoch number
- Validation metrics


## **Model Architecture**

- Faster R-CNN
- Backbone: ResNet-50 + FPN
- Pretrained on COCO (optional)
- Custom classifier head for Open Images classes
