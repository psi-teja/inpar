from layoutLMv3_utils import CustomDataset_DP
import os
from torch.utils.data import DataLoader
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Config
from sklearn.metrics import precision_recall_fscore_support
import torch.nn as nn
import torch
from tqdm import tqdm
from layoutLMv3_utils import colors
import pandas as pd


######################   starting evaluation of the best model saved    ####################


current_dir = os.path.dirname(__file__)
job_dir = os.path.join(current_dir, "results/runs/20240528084147")

dataset_folder = os.path.join(current_dir, "datasets", "imerit")

model_config = LayoutLMv3Config.from_pretrained(os.path.join(job_dir, "saved_model"))
model = LayoutLMv3ForTokenClassification(model_config)
model.load_state_dict(torch.load(os.path.join(job_dir,"saved_model", "model.pth"), map_location=torch.device('cpu')))


label2id = model_config.label2id
id2label = model_config.id2label
label_list = list(model_config.label2id.keys())


test_image_folder = os.path.join(dataset_folder, "test", "images")
test_annotation_folder = os.path.join(dataset_folder, "test", "label_json")

test_dataset = CustomDataset_DP(
    test_image_folder, test_annotation_folder, label2id=label2id
)
test_batch_size = 4


test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)


# If using GPU, move the model to GPU
model.eval()
all_preds = []
all_labels = []


with torch.no_grad():
    for batch, (inputs, labels) in enumerate(tqdm(test_dataloader, unit=f"{test_batch_size} samples")):
        output = model(**inputs)

        # Flatten predictions and labels
        preds_flat = output.logits.argmax(-1).view(-1)
        labels_flat = labels.view(-1)

        # Filter out -100 labels
        mask = labels_flat != -100
        preds_flat = preds_flat[mask]
        labels_flat = labels_flat[mask]

        for p, l in zip(preds_flat.tolist(), labels_flat.tolist()):
            all_preds.append(id2label[p])
            all_labels.append(id2label[l])


# Compute precision, recall, and F1 score
precision, recall, f1, support = precision_recall_fscore_support(
    all_labels, all_preds, labels=label_list, average=None
)


# Store the metrics as a CSV file
metrics_dict = {
    "Field": label_list,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "support": support,
}

metrics_df = pd.DataFrame(metrics_dict)
metrics_df.to_excel(os.path.join(job_dir, f"test_metric.xlsx"), index=False)
print(f"{colors.GREEN}Evaluation metrics saved successfully.{colors.END}")