**## Faster R-CNN Object Detection on Open Images**

This repository contains a PyTorch implementation of Faster R-CNN trained and evaluated on the Open Images Dataset.
The project was originally developed in Google Colab and later refactored into a modular, reusable, and GitHub-friendly structure.

The code supports:
- Training Faster R-CNN on Open Images bounding box annotations
- Validation using mAP (Mean Average Precision)
- Inference and visualization on custom images
- Checkpoint saving and loading


**## Requirements**

Create a Python environment (recommended: Python ≥ 3.9) and install dependencies.
Main dependencies:
- torch
- torchvision
- torchmetrics
- pandas
- Pillow
- matplotlib
- tqdm
