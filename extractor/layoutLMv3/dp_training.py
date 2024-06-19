import os
import sys
import subprocess
from tqdm import tqdm
import pandas as pd
import configparser
import time
import torch
from sklearn.metrics import precision_recall_fscore_support
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Config
from layoutLMv3_utils import (
    CustomDataset_DP,
    colors,
    device,
)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
current_dir = os.path.dirname(__file__)

# Check if the script is running under nohup
if not os.getenv("NOHUP"):
    # Restart the script using nohup
    nohup_command = ["nohup", sys.executable] + sys.argv + ["&"]
    # Create an environment variable to avoid recursive restarts
    new_env = os.environ.copy()
    new_env["NOHUP"] = "1"

    current_time = time.localtime()
    job_id = time.strftime("%Y%m%d%H%M%S", current_time)
    job_dir = os.path.join(current_dir, "results", "runs", f"{job_id}")

    save_model_folder = os.path.join(job_dir, "saved_model")

    print(f"{colors.GREEN}job_id: {job_id} Created!{colors.END}")
    if not os.path.exists(job_dir):
        os.makedirs(job_dir)
        os.mkdir(save_model_folder)

    new_env["JOB_DIR"] = job_dir

    # Use subprocess to start the new process
    nohup_output_file = os.path.join(job_dir, "nohup.out")
    subprocess.Popen(nohup_command, env=new_env, stdout=open(nohup_output_file, "a"), stderr=subprocess.STDOUT)
    print(f"Restarting script with nohup: {' '.join(nohup_command)}")
    sys.exit(0)


import time

start = time.time()


dataset_folder = os.path.join(current_dir, "datasets", "imerit")
print(f"Dataset Folder: {dataset_folder}")
dataset_details_file = os.path.join(dataset_folder, "details.cfg")
dataset_config = configparser.ConfigParser()
dataset_config.optionxform = str
dataset_config.read(dataset_details_file)
num_samples = dataset_config["General"]["NumberOfSamples"]

dataset_config["General"]["Dataset Folder"] = dataset_folder

label_counts = {
    label: int(count) for label, count in dataset_config["LabelCounts"].items()
}

label_list = list(label_counts.keys())
id2label = {k: v for k, v in enumerate(label_list)}
label2id = {v: k for k, v in enumerate(label_list)}

train_image_folder = os.path.join(dataset_folder, "train", "images")
train_annotation_folder = os.path.join(dataset_folder, "train", "label_jsons")
validation_image_folder = os.path.join(dataset_folder, "val", "images")
validation_annotation_folder = os.path.join(dataset_folder, "val", "label_jsons")
test_image_folder = os.path.join(dataset_folder, "test", "images")
test_annotation_folder = os.path.join(dataset_folder, "test", "label_jsons")

train_dataset = CustomDataset_DP(
    train_image_folder, train_annotation_folder, label2id=label2id
)
validation_dataset = CustomDataset_DP(
    validation_image_folder, validation_annotation_folder, label2id=label2id
)
test_dataset = CustomDataset_DP(
    test_image_folder, test_annotation_folder, label2id=label2id
)

train_size = len(train_dataset)
validation_size = len(validation_dataset)
test_size = len(test_dataset)

############################## Starting training job ####################################
continuing_from_job = False
continuing_from_job_dir = os.path.join(current_dir, "results", "runs", "20240604194714")

if continuing_from_job:
    model_config = LayoutLMv3Config.from_pretrained(
        os.path.join(continuing_from_job_dir, "saved_model")
    )
    model = LayoutLMv3ForTokenClassification(model_config)
    model.load_state_dict(
        torch.load(os.path.join(continuing_from_job_dir, "saved_model", "model.pth"), map_location=torch.device('cpu'))
    )

    loss_file_path = os.path.join(continuing_from_job_dir, "loss_history.xlsx")
    if os.path.exists(loss_file_path):
        loss_log_df = pd.read_excel(loss_file_path)
        if loss_log_df.tail(1).to_dict(orient="records"):
            epoch_start = loss_log_df.tail(1).to_dict(orient="records")[0]["Epoch"] + 1

else:
    model_path = "microsoft/layoutlmv3-base"
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        model_path, id2label=id2label, label2id=label2id
    )
    loss_log_df = pd.DataFrame(columns=["Epoch", "Train Loss", "Validation Loss"])
    epoch_start = 1

model.to(device)
model = nn.DataParallel(model)
job_dir = os.getenv('JOB_DIR')
loss_file_path = os.path.join(job_dir, "loss_history.xlsx")

best_loss = float("inf")
patience = 5  # Number of epochs to wait for improvement before stopping
epochs_no_improve = 0

learning_rate = 1e-4
train_batch_size = 16
epochs = 120
training_config = configparser.ConfigParser()
training_config.optionxform = str

training_config["TrainingDetails"] = {
    "DatasetFolder": dataset_folder,
    "TrainSamples": str(len(train_dataset)),
    "ValSamples": str(len(validation_dataset)),
    "TestSamples": str(len(test_dataset)),
    "LearningRate": str(learning_rate),
    "BatchSize": str(train_batch_size),
    "Epochs": str(epochs + epoch_start - 1),
}
with open(os.path.join(job_dir, "training_details.cfg"), "w") as configfile:
    training_config.write(configfile)
    print(
        f"{colors.GREEN}Training details stored{colors.END}"
    )

train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=train_batch_size, shuffle=False)

loss_fn = nn.CrossEntropyLoss()
classifier_optimizer = optim.Adam(
    model.module.classifier.parameters(), lr=learning_rate
)

for param in model.module.base_model.parameters():
    param.requires_grad = False  # freezing pretrained model weights


for epoch in tqdm(range(epoch_start, epoch_start+epochs), unit="epoch"):
    epoch_loss = 0
    model.train()
    for batch, (inputs, labels) in enumerate(train_dataloader):
        classifier_optimizer.zero_grad()
        output = model(**inputs)

        preds_logits_flat = output.logits.view(-1, output.logits.shape[-1])
        labels_flat = labels.view(-1)

        mask = labels_flat != -100
        preds_logits_flat = preds_logits_flat[mask]
        labels_flat = labels_flat[mask]

        loss = loss_fn(preds_logits_flat, labels_flat)
        loss.backward()
        classifier_optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(train_dataloader)

    # Compute validation loss
    model.eval()
    validation_loss = 0

    with torch.no_grad():
        for batch, (inputs, labels) in enumerate(validation_dataloader):
            output = model(**inputs)

            preds_logits_flat = output.logits.view(-1, output.logits.shape[-1])
            labels_flat = labels.view(-1)

            mask = labels_flat != -100
            preds_logits_flat = preds_logits_flat[mask]
            labels_flat = labels_flat[mask]

            loss = loss_fn(preds_logits_flat, labels_flat)
            validation_loss += loss.item()

    validation_loss /= len(validation_dataloader)

    # Log the losses
    new_row = {
        "Epoch": epoch,
        "Train Loss": epoch_loss,
        "Validation Loss": validation_loss,
    }

    # Create a DataFrame from new_row with a single index [0]
    new_row_df = pd.DataFrame(new_row, index=[0])

    # Concatenate the new row DataFrame with the existing loss_log_df
    loss_log_df = pd.concat([loss_log_df, new_row_df], ignore_index=True)

    # Save the log DataFrame to Excel file
    loss_log_df.to_excel(loss_file_path, index=False)

    # Check for improvement
    if validation_loss < best_loss:
        best_loss = validation_loss
        epochs_no_improve = 0
        torch.save(
            model.module.state_dict(), os.path.join(job_dir, "saved_model", "model.pth")
        )
        model_config = model.module.config
        model_config.save_pretrained(os.path.join(job_dir, "saved_model"))
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + epoch_start}")
            break

print(f"{colors.GREEN}Training completed and best model saved successfully.{colors.END}")

######################   starting evaluation of the best model saved    ####################
# job_dir = "AIBackend/DocAI/inpar-research/layoutLMv3/results/runs/20240417191141/saved_model"

test_batch_size = 16
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

model_config = LayoutLMv3Config.from_pretrained(os.path.join(job_dir, "saved_model"))
model = LayoutLMv3ForTokenClassification(model_config)

# Load model's state dictionary
model.load_state_dict(torch.load(os.path.join(job_dir, "saved_model", "model.pth")))

# If using GPU, move the model to GPU
model.to(device)
model = nn.DataParallel(model)
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch, (inputs, labels) in enumerate(
        tqdm(test_dataloader, unit=f"{test_batch_size} samples")
    ):
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
metrics_df.to_excel(os.path.join(job_dir, f"test_classification_report.xlsx"), index=False)
print(f"{colors.GREEN}Evaluation metrics saved successfully.{colors.END}")

end = time.time()

elapsed_time_seconds = end - start
elapsed_time_hours = elapsed_time_seconds / 3600

print(f"{colors.GREEN}Training time: {elapsed_time_hours} hours{colors.END}")
