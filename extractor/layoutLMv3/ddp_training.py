import os
import time
import torch
import argparse
import configparser
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Config
import torch.multiprocessing as mp
from layoutLMv3_utils import CustomDataset_DDP, colors
from tqdm import tqdm

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Trainer:
    def __init__(self, model, train_data, val_data, test_data, optimizer, gpu_id, job_dir, patience, epoch_start, loss_log_df, id2label, label2id):
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.model = DDP(model, device_ids=[gpu_id])
        self.job_dir = job_dir
        self.train_loss_history = []
        self.val_loss_history = []
        self.patience = patience
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.epoch_start = epoch_start
        self.loss_log_df = loss_log_df
        self.id2label = id2label
        self.label2id = label2id

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(**source)
        preds_logits_flat = output.logits.view(-1, output.logits.shape[-1])
        labels_flat = targets.view(-1)
        mask = labels_flat != -100
        preds_logits_flat = preds_logits_flat[mask]
        labels_flat = labels_flat[mask]
        loss = torch.nn.CrossEntropyLoss()(preds_logits_flat, labels_flat)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch, max_epochs):
        self.train_data.sampler.set_epoch(epoch)
        epoch_loss = 0.0
        num_batches = len(self.train_data)
        progress_bar = tqdm(enumerate(self.train_data), total=num_batches, desc=f"Epoch {epoch + self.epoch_start}/{max_epochs}")
        for batch_idx, (source, targets) in progress_bar:
            targets = targets.to(self.gpu_id)
            loss = self._run_batch(source, targets)
            epoch_loss += loss
            progress_bar.set_postfix({"loss": loss})
        avg_epoch_loss = epoch_loss / num_batches
        if self.gpu_id == 0:
            self.train_loss_history.append(avg_epoch_loss)

    def _save_checkpoint(self):
        if self.job_dir:
            save_model_folder = os.path.join(self.job_dir, "saved_model")
            os.makedirs(save_model_folder, exist_ok=True)
            torch.save(self.model.module.state_dict(), os.path.join(save_model_folder, "model.pth"))
            model_config = self.model.module.config
            model_config.save_pretrained(save_model_folder)

    def evaluate(self, data_loader, filename_prefix):
        self.model.eval()
        y_true, y_pred = [], []
        epoch_loss = 0.0
        num_batches = len(data_loader)
        with torch.no_grad():
            for source, targets in data_loader:
                model_input = {
                    "input_ids": source["input_ids"].to(self.gpu_id),
                    "attention_mask": source["attention_mask"].to(self.gpu_id),
                    "bbox": source["bbox"].to(self.gpu_id),
                    "pixel_values": source["pixel_values"].to(self.gpu_id),
                }
                targets = targets.to(self.gpu_id)
                output = self.model(**model_input)
                preds = torch.argmax(output.logits, dim=-1)
                for t, p in zip(targets.view(-1), preds.view(-1)):
                    if t != -100:
                        y_true.append(self.id2label[t.item()])
                        y_pred.append(self.id2label[p.item()])
                preds_logits_flat = output.logits.view(-1, output.logits.shape[-1])
                labels_flat = targets.view(-1)
                mask = labels_flat != -100
                preds_logits_flat = preds_logits_flat[mask]
                labels_flat = labels_flat[mask]
                loss = torch.nn.CrossEntropyLoss()(preds_logits_flat, labels_flat)

                epoch_loss += loss.item()
        avg_epoch_loss = epoch_loss / num_batches
        # Compute precision, recall, and F1 score

        label_list = list(self.label2id.keys())

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=label_list, average=None
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
        metrics_df.to_excel(os.path.join(self.job_dir, f"{filename_prefix}_classification_report.xlsx"), index=False)

        cm = confusion_matrix(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        f1_df = pd.DataFrame({"F1 Score": [f1]})
        f1_df.to_excel(os.path.join(self.job_dir, f"{filename_prefix}_f1_score.xlsx"), index=False)
        plt.figure(figsize=(50, 40))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{filename_prefix.capitalize()} Confusion Matrix")
        plt.savefig(os.path.join(self.job_dir, f"{filename_prefix}_confusion_matrix.png"))
        return f1, avg_epoch_loss

    def save_loss_history(self):
        new_row = {
            "Epoch": list(range(self.epoch_start, len(self.train_loss_history) + self.epoch_start)),
            "Train Loss": self.train_loss_history,
            "Validation Loss": self.val_loss_history
        }

        loss_df = pd.concat([self.loss_log_df, pd.DataFrame(new_row)], ignore_index=True)
        loss_df.to_excel(os.path.join(self.job_dir, "loss_history.xlsx"), index=False)

    def train(self, max_epochs):
        for epoch in range(max_epochs):
            self._run_epoch(epoch, max_epochs)
            if self.gpu_id == 0:  
                val_f1, val_loss = self.evaluate(self.val_data, "val")
                self.val_loss_history.append(val_loss)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint()
                    self.epochs_no_improve = 0
                else:
                    self.epochs_no_improve += 1
                    if self.epochs_no_improve >= self.patience:
                        print(f"{colors.GREEN}Early stopping at epoch {epoch} {colors.END}")
                        break
                self.save_loss_history()

    def evaluate_test(self):
        self.evaluate(self.test_data, "test")


def load_train_objs(
    train_image_folder,
    train_annotation_folder,
    val_image_folder,
    val_annotation_folder,
    test_image_folder,
    test_annotation_folder,
    label2id,
    id2label,
    current_dir
):
    train_dataset = CustomDataset_DDP(train_image_folder, train_annotation_folder, label2id=label2id)
    val_dataset = CustomDataset_DDP(val_image_folder, val_annotation_folder, label2id=label2id)
    test_dataset = CustomDataset_DDP(test_image_folder, test_annotation_folder, label2id=label2id)
    
    continuing_from_job = False
    continuing_from_job_dir = os.path.join(current_dir, "results", "runs", "20240507150135")

    if continuing_from_job:
        model_config = LayoutLMv3Config.from_pretrained(
            os.path.join(continuing_from_job_dir, "saved_model")
        )
        model = LayoutLMv3ForTokenClassification(model_config)
        model.load_state_dict(
            torch.load(os.path.join(continuing_from_job_dir, "saved_model", "model.pth"))
        )

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
        epoch_start = 1
        loss_log_df = pd.DataFrame(columns=["Epoch", "Train Loss", "Validation Loss"])
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-4)
    return train_dataset, val_dataset, test_dataset, model, optimizer, epoch_start, loss_log_df

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        shuffle=False,
        sampler=DistributedSampler(dataset),
    )

def main(
    rank: int,
    world_size: int,
    total_epochs: int,
    batch_size: int,
    train_image_folder,
    train_annotation_folder,
    val_image_folder,
    val_annotation_folder,
    test_image_folder,
    test_annotation_folder,
    label2id,
    id2label,
    job_dir,
    patience: int,
    current_dir,
):
    ddp_setup(rank, world_size)
    train_dataset, val_dataset, test_dataset, model, optimizer, epoch_start, loss_log_df = load_train_objs(
        train_image_folder,
        train_annotation_folder,
        val_image_folder,
        val_annotation_folder,
        test_image_folder,
        test_annotation_folder,
        label2id,
        id2label,
        current_dir
    )


    train_data = prepare_dataloader(train_dataset, batch_size)
    val_data = prepare_dataloader(val_dataset, batch_size)
    test_data = prepare_dataloader(test_dataset, batch_size)
    trainer = Trainer(
        model, train_data, val_data, test_data, optimizer, rank, job_dir, patience, epoch_start, loss_log_df, id2label, label2id
    )
    trainer.train(total_epochs)
    trainer.evaluate_test()

if __name__ == "__main__":
    import time

    start = time.time()

    current_dir = os.path.dirname(__file__)
    dataset_folder = os.path.join(current_dir, "datasets", "imerit")
    dataset_details_file = os.path.join(dataset_folder, "details.cfg")
    dataset_config = configparser.ConfigParser()
    dataset_config.optionxform = str
    dataset_config.read(dataset_details_file)
    label_counts = {
        label: int(count) for label, count in dataset_config["LabelCounts"].items()
    }
    label_list = list(label_counts.keys())
    id2label = {k: v for k, v in enumerate(label_list)}
    label2id = {v: k for k, v in enumerate(label_list)}
    train_image_folder = os.path.join(dataset_folder, "train", "images")
    train_annotation_folder = os.path.join(dataset_folder, "train", "label_jsons")
    val_image_folder = os.path.join(dataset_folder, "val", "images")
    val_annotation_folder = os.path.join(dataset_folder, "val", "label_jsons")
    test_image_folder = os.path.join(dataset_folder, "test", "images")
    test_annotation_folder = os.path.join(dataset_folder, "test", "label_jsons")
    job_id = f"{time.strftime('%Y%m%d%H%M%S')}"
    job_dir = os.path.join(current_dir, "results", "runs", job_id)
    print(f"{colors.GREEN}job_id: {job_id} Created{colors.END}")
    os.makedirs(job_dir, exist_ok=True)
    parser = argparse.ArgumentParser(description="simple distributed training job")
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=100,
        help="Total epochs to train the model"
    )
    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
        help="Input batch size on each device (default: 4)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Number of epochs to wait for improvement before early stopping (default: 5)",
    )
    args = parser.parse_args()
    world_size = torch.cuda.device_count()

    print(f"{colors.GREEN}Max Epochs: {args.epochs}{colors.END}")

    print(f"{colors.GREEN}Batch Size: {args.batch_size}{colors.END}")

    mp.spawn(
        main,
        args=(
            world_size,
            args.epochs,
            args.batch_size,
            train_image_folder,
            train_annotation_folder,
            val_image_folder,
            val_annotation_folder,
            test_image_folder,
            test_annotation_folder,
            label2id,
            id2label,
            job_dir,
            args.patience,
            current_dir
        ),
        nprocs=world_size,
    )

    end = time.time()

    elapsed_time_seconds = end - start
    elapsed_time_hours = elapsed_time_seconds / 3600

    print(f"{colors.GREEN}Training time: {elapsed_time_hours} hours{colors.END}")


