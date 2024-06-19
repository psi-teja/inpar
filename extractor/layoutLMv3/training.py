import os, tarfile, shutil
from tqdm import tqdm
import pandas as pd
import configparser
import time, torch
from sklearn.metrics import precision_recall_fscore_support
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Config
from layoutLMv3_utils import upload_folder_to_s3, CustomDataset, colors, fraction_to_ratio, device

current_dir = os.path.dirname(__file__)

dataset_folder = os.path.join(current_dir, "datasets", "all")
print(f"Dataset Folder: {dataset_folder}")
dataset_details_file = os.path.join(dataset_folder, "details.cfg")
dataset_config = configparser.ConfigParser()
dataset_config.optionxform = str
dataset_config.read(dataset_details_file)
num_samples = dataset_config["General"]["NumberOfSamples"]
print(f"NumberOfSamples: {num_samples}")

dataset_config["General"]["Dataset Folder"] = dataset_folder

label_counts = {
    label: int(count) for label, count in dataset_config["LabelCounts"].items()
}
print("Label\tcount")
for label in label_counts:
    print(label, label_counts[label])

label_list = list(label_counts.keys())
id2label = {k: v for k, v in enumerate(label_list)}
label2id = {v: k for k, v in enumerate(label_list)}

train_image_folder = os.path.join(dataset_folder,"train", "images")
train_annotation_folder = os.path.join(dataset_folder,"train", "label_json")
test_image_folder = os.path.join(dataset_folder,"test", "images")
test_annotation_folder = os.path.join(dataset_folder,"test", "label_json")

train_dataset = CustomDataset(train_image_folder, train_annotation_folder, label2id=label2id)
test_dataset = CustomDataset(test_image_folder, test_annotation_folder, label2id=label2id)
train_size = len(train_dataset)
test_size = len(test_dataset)

train_test_ratio = fraction_to_ratio(train_size, test_size)

print(f"{colors.GREEN}-------------------------- train:test ratio is {train_test_ratio}{colors.END}")

############################# create training job_id using time stamp ############################
current_time = time.localtime()
job_id = time.strftime("%Y%m%d%H%M%S", current_time)
job_dir = os.path.join(current_dir, "results", "runs", f"{job_id}")


save_model_folder = os.path.join(job_dir, "saved_model")
if not os.path.exists(job_dir):
    os.makedirs(job_dir)
    os.mkdir(save_model_folder)
print(f"{colors.GREEN}-------------------------- job_id: {job_id} Created{colors.END}")


model_bucket_name = "tally-ai-doc-ai-inpar-models" 
s3_job_dir = f"layoutLMv3_finetuned/{job_id}"

############################## Starting training job ####################################
continuing_from_job = False
continuing_from_job_dir = os.path.join(current_dir, "results", "runs","20240426114909")

if continuing_from_job:
    model_config = LayoutLMv3Config.from_pretrained(os.path.join(continuing_from_job_dir, "saved_model"))
    model = LayoutLMv3ForTokenClassification(model_config)
    model.load_state_dict(torch.load(os.path.join(continuing_from_job_dir,"saved_model", "model.pth")))
    
    loss_file_path = os.path.join(continuing_from_job_dir, "loss.xlsx")
    if os.path.exists(loss_file_path):
        loss_log_df = pd.read_excel(loss_file_path)
        if loss_log_df.tail(1).to_dict(orient="records"):
            epoch_start = loss_log_df.tail(1).to_dict(orient="records")[0]["epoch"] + 1

else:
    model_path = "microsoft/layoutlmv3-base"
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        model_path, id2label=id2label, label2id=label2id
    )
    loss_log_df = pd.DataFrame(columns=["epoch", "loss"])
    epoch_start = 1

model.to(device)
model = nn.DataParallel(model)
model.train()
loss_writer = pd.ExcelWriter(os.path.join(job_dir, "loss.xlsx"), engine="xlsxwriter")

best_loss = float("inf")

learning_rate = 1e-4
train_batch_size = 128
epochs = 1
training_config = configparser.ConfigParser()
training_config.optionxform = str

training_config["TrainingDetails"] = {
    "DatasetFolder": dataset_folder,
    "TrainTestRatio": str(train_test_ratio),
    "LearningRate": str(learning_rate),
    "BatchSize": str(train_batch_size),
    "Epochs": str(epochs + epoch_start - 1)
}
with open(os.path.join(job_dir, "training_details.cfg"), "w") as configfile:
    training_config.write(configfile)
    print(f"{colors.GREEN}--------------------------Training details stored{colors.END}")


train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)


loss_fn = nn.CrossEntropyLoss()
classifier_optimizer = optim.Adam(model.module.classifier.parameters(), lr=learning_rate)

for param in model.module.base_model.parameters():
    param.requires_grad = False  # freezing pretrained model weights

for epoch in tqdm(range(epochs), unit="epoch"):

    epoch_loss = 0

    for batch, (inputs, labels) in enumerate(train_dataloader):
        classifier_optimizer.zero_grad()
        output = model(**inputs)

        preds_logits_flat = output.logits.view(-1, output.logits.shape[-1])
        labels_flat = labels.view(-1)

        mask = labels_flat != -100
        preds_logits_flat = preds_logits_flat[mask]
        labels_flat = labels_flat[mask]

        # print(preds_logits_flat.get_device(), labels_flat.get_device())
        loss = loss_fn(preds_logits_flat, labels_flat)
        loss.backward()
        classifier_optimizer.step()

        epoch_loss += loss

    if loss_log_df.empty:
        loss_log_df = pd.DataFrame(
            {"epoch": [epoch + epoch_start], "loss": [epoch_loss.detach().item()]}
        )
    else:
        loss_log_df = pd.concat(
            [
                loss_log_df,
                pd.DataFrame(
                    {
                        "epoch": [epoch + epoch_start],
                        "loss": [epoch_loss.detach().item()],
                    }
                ),
            ],
            ignore_index=True,
        )

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.module.state_dict(), os.path.join(job_dir,"saved_model", "model.pth")) 
        model_config = model.module.config
        model_config.save_pretrained(os.path.join(job_dir, "saved_model"))

loss_log_df.to_excel(loss_writer, index=False)
loss_writer._save()

######################   starting evaluation of the best model saved    ####################
# job_dir = "AIBackend/DocAI/inpar-research/layoutLMv3/results/runs/20240417191141/saved_model"

test_batch_size = 32
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

model_config = LayoutLMv3Config.from_pretrained(os.path.join(job_dir, "saved_model"))
model = LayoutLMv3ForTokenClassification(model_config)

# Load model's state dictionary
model.load_state_dict(torch.load(os.path.join(job_dir,"saved_model", "model.pth")))

# If using GPU, move the model to GPU
model.to(device)
model = nn.DataParallel(model)
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


# tar_filename = os.path.join(job_dir, "saved_model.tar.gz")
# with tarfile.open(tar_filename, "w:gz") as tar:
#     for filename in os.listdir(os.path.join(job_dir, "saved_model")):
#         tar.add(os.path.join(job_dir, "saved_model", filename), arcname=os.path.basename(os.path.join(job_dir, "saved_model", filename)))
# shutil.rmtree(os.path.join(job_dir, "saved_model"))
# upload_folder_to_s3(job_dir, model_bucket_name, s3_job_dir)
# print(f"{colors.GREEN}Saved Model artifacts is uploaded to S3{colors.END}")


