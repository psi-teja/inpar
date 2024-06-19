# LayoutLMv3 finetuning Pipeline

This repository contains scripts and instructions to finetune a LayoutLMv3 model for token classification using custom datasets.

## Setup

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Steps to Run

Follow the steps below to run the scripts in the specified order:

### 1. Download Individual Fields Data

Use the `download_individual_fields_safehouse.py` script to download the Individual Fields Data from Safehouse.

```bash
python download_individual_fields_safehouse.py
```

### 2. Data Preparation

Use the `prepare_data.py` script to generate TallyAI Json for each and every page
```bash
python prepare_data.py
```

### 3. Process and label images

Use the `process_and_label_images.py` script to generate TallyAI Json for each and every page
```bash
python process_and_label_images.py
```

### 4. Split Dataset

Use the `split_dataset.py` script to split dataset into train valication and test datasets
```bash
python split_dataset.py
```


### 3. Model Training

```bash
# On CPU
python training.py
```

```bash
# DP training
python dp_training.py
```

```bash
# DDP training
python ddp_training.py
```


Training results are stored in `results/runs/exp{i}`, where i is a sequential number starting from 1.