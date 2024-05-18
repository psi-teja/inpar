import torch, time
from torch.utils.data import Dataset, DataLoader
from utils import CustomDataset_DDP
import configparser
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Config

import os


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])

        # Create job_dir using the common job_id
        self.job_dir = None
        if gpu_id == 0:
            current_time = time.localtime()
            current_dir = os.path.dirname(__file__)
            job_id = time.strftime("%Y%m%d%H%M%S", current_time)
            self.job_dir = os.path.join(current_dir, "results", "runs", f"{job_id}")
            save_model_folder = os.path.join(self.job_dir, "saved_model")
            if not os.path.exists(self.job_dir):
                os.makedirs(self.job_dir)
                os.mkdir(save_model_folder)
                print(f"Created job directory: {self.job_dir}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(**source)

        preds_logits_flat = output.logits.view(-1, output.logits.shape[-1])
        labels_flat = targets.view(-1)

        mask = labels_flat != -100
        preds_logits_flat = preds_logits_flat[mask]
        labels_flat = labels_flat[mask]

        # print(preds_logits_flat.get_device(), labels_flat.get_device())
        loss = torch.nn.CrossEntropyLoss()(preds_logits_flat, labels_flat)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(
            f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}"
        )
        self.train_data.sampler.set_epoch(epoch)

        for source, targets in self.train_data:
            model_input = {
                "input_ids": source["input_ids"].to(self.gpu_id),
                "attention_mask": source["attention_mask"].to(self.gpu_id),
                "bbox": source["bbox"].to(self.gpu_id),
                "pixel_values": source["pixel_values"].to(self.gpu_id),
            }
            targets = targets.to(self.gpu_id)
            self._run_batch(model_input, targets)

    def _save_checkpoint(self, epoch):
        if self.job_dir is not None:
            save_model_folder = os.path.join(self.job_dir, "saved_model")
            torch.save(
                self.model.module.state_dict(),
                os.path.join(self.job_dir, "saved_model", "model.pth"),
            )
            model_config = self.model.module.config
            model_config.save_pretrained(save_model_folder)

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs(train_image_folder, train_annotation_folder, label2id, id2label):
    train_dataset = CustomDataset_DDP(
        train_image_folder, train_annotation_folder, label2id=label2id
    )
    model_path = "microsoft/layoutlmv3-base"
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        model_path, id2label=id2label, label2id=label2id
    )
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-4)
    return train_dataset, model, optimizer


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
    save_every: int,
    total_epochs: int,
    batch_size: int,
    train_image_folder,
    train_annotation_folder,
    label2id,
    id2label,
):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs(
        train_image_folder, train_annotation_folder, label2id, id2label
    )
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse

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

    train_image_folder = os.path.join(dataset_folder, "train", "images")
    train_annotation_folder = os.path.join(dataset_folder, "train", "label_json")

    parser = argparse.ArgumentParser(description="simple distributed training job")
    parser.add_argument(
        "total_epochs", type=int, help="Total epochs to train the model"
    )
    parser.add_argument("save_every", type=int, help="How often to save a snapshot")
    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
        help="Input batch size on each device (default: 32)",
    )
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(
        main,
        args=(
            world_size,
            args.save_every,
            args.total_epochs,
            args.batch_size,
            train_image_folder,
            train_annotation_folder,
            label2id,
            id2label,
        ),
        nprocs=world_size,
    )
