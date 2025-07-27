import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any
from tqdm import tqdm

from base_trainer import BaseTrainer
from model import PointNetClassification
from dataloader import get_data_loaders
from utils.loss import Accuracy


class ClassificationTrainer(BaseTrainer):
    """
    Trainer class for PointNet classification on ModelNet40 dataset.

    Extends BaseTrainer to implement the specifics for
    classification task including model building, data loaders,
    optimizer, scheduler, and training/validation/test steps.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing
            training parameters like 'epochs', 'batch_size', 'lr', 'gpu', etc.
    """

    def build_model(self) -> torch.nn.Module:
        """
        Builds and returns the PointNetCls classification model.

        Returns:
            torch.nn.Module: The PointNetCls model instance.
        """
        return PointNetClassification(num_classes=40, input_transform=True, feature_transform=True)

    def build_dataloaders(self) -> Dict[str, DataLoader]:
        """
        Creates train, validation, and test data loaders using your PointCloudDataset.
        """
        _, dataloaders = get_data_loaders(
            dataset_name="modelnet",
            data_dir="./data",
            batch_size=self.config["batch_size"],
            phases=["train", "val", "test"],
            from_folder=False,
        )
        phase_names = ["train", "val", "test"]
        return dict(zip(phase_names, dataloaders))


    def build_optimizer(self) -> torch.optim.Optimizer:
        """
        Creates and returns the Adam optimizer for model parameters.

        Returns:
            torch.optim.Optimizer: Adam optimizer instance.
        """
        return torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])

    def build_scheduler(self):
        """
        Creates and returns a MultiStepLR scheduler.

        Returns:
            torch.optim.lr_scheduler._LRScheduler: Learning rate scheduler.
        """
        return torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[30, 80], gamma=0.5
        )

    def checkpoint_metric_name(self) -> str:
        """
        Returns the metric name used to monitor checkpointing.

        Returns:
            str: Name of the metric to track ('val_acc').
        """
        return "val_acc"

    def checkpoint_mode(self) -> str:
        """
        Returns the mode for the checkpoint metric optimization.

        Returns:
            str: 'max' because we want to maximize accuracy.
        """
        return "max"

    def step(self, points: torch.Tensor, labels: torch.Tensor):
        """
        Forward pass and loss computation for a batch.

        Args:
            points (torch.Tensor): Input point cloud tensor of shape [B, N, 3].
            labels (torch.Tensor): Ground truth labels tensor of shape [B].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Loss tensor and predicted class indices.
        """
        points = points.to(self.device)
        labels = labels.to(self.device)
        logits = self.model(points)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        return loss, preds

    def train_one_epoch(self, epoch: int) -> None:
        """
        Executes one full training epoch.

        Args:
            epoch (int): Current epoch number.
        """
        self.model.train()
        self.train_acc_metric = Accuracy()
        running_loss = []

        pbar = tqdm(self.loaders["train"], desc=f"Epoch {epoch+1}/{self.config['epochs']} - Training")
        for points, labels in pbar:
            self.optimizer.zero_grad()
            loss, preds = self.step(points, labels)
            loss.backward()
            self.optimizer.step()

            running_loss.append(loss.item())
            batch_acc = self.train_acc_metric(preds, labels.to(self.device))
            pbar.set_postfix(loss=loss.item(), accuracy=f"{batch_acc*100:.1f}%")

        avg_loss = sum(running_loss) / len(running_loss)
        epoch_acc = self.train_acc_metric.compute_epoch()
        print(f"Train Epoch {epoch+1}: Loss {avg_loss:.4f} | Accuracy {epoch_acc*100:.1f}%")

    def validate(self, epoch: int) -> float:
        """
        Runs validation over the validation dataset.

        Args:
            epoch (int): Current epoch number.

        Returns:
            float: Validation accuracy used for checkpointing.
        """
        self.model.eval()
        self.val_acc_metric = Accuracy()
        running_loss = []

        with torch.no_grad():
            for points, labels in self.loaders["val"]:
                points, labels = points.to(self.device), labels.to(self.device)
                loss, preds = self.step(points, labels)
                running_loss.append(loss.item())
                self.val_acc_metric(preds, labels)

        avg_loss = sum(running_loss) / len(running_loss)
        epoch_acc = self.val_acc_metric.compute_epoch()
        print(f"Validation Epoch {epoch+1}: Loss {avg_loss:.4f} | Accuracy {epoch_acc*100:.1f}%")

        return epoch_acc.item()

    def test(self) -> None:
        """
        Runs evaluation on the test dataset and prints test accuracy.
        """
        self.model.eval()
        self.test_acc_metric = Accuracy()

        with torch.no_grad():
            for points, labels in self.loaders["test"]:
                points, labels = points.to(self.device), labels.to(self.device)
                _, preds = self.step(points, labels)
                self.test_acc_metric(preds, labels)

        test_acc = self.test_acc_metric.compute_epoch()
        print(f"Test Accuracy: {test_acc*100:.1f}%")
