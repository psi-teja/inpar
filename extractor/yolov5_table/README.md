# YOLOv5 ROI Training Pipeline

This repository contains scripts and instructions to train a YOLOv5 model for Region of Interest (ROI) detection using custom datasets.

## Setup

1. Clone this repository:
    ```bash
    cd AIBackend/DocAI/inpar-research/extractor/yolov5_table
    git clone https://github.com/ultralytics/yolov5.git
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Steps to Run

Follow the steps below to run the scripts in the specified order:

### 1. Generate Ground Truth

Use the `../generate_row_col_dect_gt.py` script to download the ROI data from Safehouse.

```bash
python ../generate_row_col_dect_gt.py
```

### 2. Data Preparation

Use the `prepare_data.py` script to pre-process

```bash
python prepare_data.py
```

### 3. Data Preparation

Use the `split_dataset.py` script to pre-process

```bash
python split_dataset.py
```

### 4. Model Training

```bash
# On CPU
python yolov5/train.py --data datasets/imerit/row_col.yml --weights yolov5s.pt --img 640 --project results/runs
```

```bash
# DDP training on 8 NVIDIA A10G GPUs machine
python -m torch.distributed.run --nproc_per_node 8 yolov5/train.py --batch 64 --data datasets/imerit/row_cal.yml --weights yolov5s.pt --project results/runs --device 0,1,2,3,4,5,6,7
```

```bash
# DP training on 1 NVIDIA T1000 GPU machine
python yolov5/train.py --img 640 --batch 16 --epochs 50 --data datasets/imerit/row_col.yml --weights yolov5s.pt --project results/runs --device 0
# Run the training script with nohup
nohup python yolov5/train.py --img 640 --batch 16 --epochs 50 --data datasets/imerit/row_col.yml --weights yolov5s.pt --project results/runs --device 0 > temp_train_output.log 2>&1 &

```

Training results are stored in `results/runs/exp{i}`, where i is a dynamically generated sequential number