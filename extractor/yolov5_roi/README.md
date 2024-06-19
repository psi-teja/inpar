# YOLOv5 ROI Training Pipeline

This repository contains scripts and instructions to train a YOLOv5 model for Region of Interest (ROI) detection using custom datasets.

## Setup

1. Clone this repository:
    ```bash
    cd AIBackend/DocAI/inpar-research/extractor/yolov5_roi
    git clone https://github.com/ultralytics/yolov5.git
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Steps to Run

Follow the steps below to run the scripts in the specified order:

### 1. Download ROI Data

Use the `download_roi_safehouse.py` script to download the ROI data from Safehouse.

```bash
python download_roi_safehouse.py
```

### 2. Data Preparation

Use the `data_prep.py` script to pre-process and split data into train, test, val

```bash
python data_prep.py
```

### 3. Model Training

```bash
# On CPU
python yolov5/train.py --data datasets/imerit/roi.yml --weights yolov5s.pt --img 640 --project results/runs
```

```bash
# DDP training on 8 NVIDIA A10G GPUs machine
python -m torch.distributed.run --nproc_per_node 8 yolov5/train.py --batch 64 --data datasets/imerit/roi.yml --weights yolov5s.pt --project results/runs --device 0,1,2,3,4,5,6,7
```

```bash
# DP training on 1 NVIDIA T1000 GPU machine
python yolov5/train.py --img 640 --batch 16 --epochs 50 --data datasets/imerit/roi.yml --weights yolov5s.pt --project results/runs --device 0
```


Training results are stored in `results/runs/exp{i}`, where i is a sequential number starting from 1.