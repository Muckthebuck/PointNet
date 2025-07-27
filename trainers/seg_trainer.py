from typing import Dict, Any
import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer
from tqdm import tqdm

from base_trainer import BaseTrainer
from model import PointNetPartSegmentation
from dataloader import get_data_loaders
from utils.loss import Accuracy, mIoU

class SegmentationTrainer(BaseTrainer):
    """
    Trainer class for PointNet Part Segmentation task.

    Implements all abstract methods from BaseTrainer for
    segmentation-specific model, dataloaders, training and validation logic.

    Args:
        config (Dict[str, Any]): Configuration dictionary with keys like
            'epochs', 'batch_size', 'lr', 'gpu', 'save', etc.
    """

    def build_model(self) -> Module:
        """Builds and returns the PointNetPartSegmentation model."""
        return PointNetPartSegmentation()

    def build_dataloaders(self) -> Dict[str, DataLoader]:
        _, dataloaders = get_data_loaders(
            dataset_name=self.config.get("dataset_name", "modelnet"),
            data_dir=self.config.get("data_dir", "./data"),
            batch_size=self.config["batch_size"],
            phases=["train", "val", "test"],
            from_folder=self.config.get("from_folder", False),
        )
        phase_names = ["train", "val", "test"]
        return dict(zip(phase_names, dataloaders))

    def build_optimizer(self) -> Optimizer:
        """Creates the Adam optimizer for the model parameters."""
        return torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])

    def build_scheduler(self) -> Any:
        """Creates a MultiStepLR scheduler for learning rate decay."""
        return torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[30, 80], gamma=0.5
        )

    def train_one_epoch(self, epoch: int) -> None:
        """
        Performs a single training epoch on the training dataset.

        Args:
            epoch (int): The current epoch index.
        """
        self.model.train()
        train_loader = self.loaders["train"]
        train_acc_metric = Accuracy()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config['epochs']} - Training")
        for points, pc_labels, class_labels in pbar:
            points = points.to(self.device)
            pc_labels = pc_labels.to(self.device)
            class_labels = class_labels.to(self.device)

            self.optimizer.zero_grad()
            loss, logits, preds = self.step(points, pc_labels, class_labels)
            loss.backward()
            self.optimizer.step()

            train_acc_metric.update(preds, pc_labels)
            running_loss += loss.item()

            pbar.set_postfix(loss=loss.item(), accuracy=train_acc_metric.compute_epoch())

    def validate(self, epoch: int) -> float:
        """
        Runs validation for the current epoch and returns the validation metric.

        Args:
            epoch (int): The current epoch index.

        Returns:
            float: The validation metric used for checkpointing (mean IoU).
        """
        self.model.eval()
        val_loader = self.loaders["val"]
        val_acc_metric = Accuracy()
        val_iou_metric = mIoU()
        val_loss = 0.0

        with torch.no_grad():
            for points, pc_labels, class_labels in val_loader:
                points = points.to(self.device)
                pc_labels = pc_labels.to(self.device)
                class_labels = class_labels.to(self.device)

                loss, logits, preds = self.step(points, pc_labels, class_labels)

                val_acc_metric.update(preds, pc_labels)
                val_iou_metric.update(logits, pc_labels, class_labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou_metric.compute_epoch()

        print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f} | mIoU: {avg_val_iou:.4f}")

        return avg_val_iou.item()

    def test(self) -> None:
        """
        Runs testing on the test dataset and prints final metrics.
        """
        self.model.eval()
        test_loader = self.loaders["test"]
        test_acc_metric = Accuracy()
        test_iou_metric = mIoU()
        test_loss = 0.0

        with torch.no_grad():
            for points, pc_labels, class_labels in test_loader:
                points = points.to(self.device)
                pc_labels = pc_labels.to(self.device)
                class_labels = class_labels.to(self.device)

                loss, logits, preds = self.step(points, pc_labels, class_labels)

                test_acc_metric.update(preds, pc_labels)
                test_iou_metric.update(logits, pc_labels, class_labels)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        avg_test_acc = test_acc_metric.compute_epoch()
        avg_test_iou = test_iou_metric.compute_epoch()

        print(f"Test Loss: {avg_test_loss:.4f} | Accuracy: {avg_test_acc:.4f} | mIoU: {avg_test_iou:.4f}")

    def step(self, points: torch.Tensor, pc_labels: torch.Tensor, class_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the segmentation model and computes loss.

        Args:
            points (torch.Tensor): Input point clouds, shape [B, N, 3].
            pc_labels (torch.Tensor): Ground truth per-point labels, shape [B, N].
            class_labels (torch.Tensor): Class labels for the point clouds, shape [B].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - loss: Scalar tensor loss.
                - logits: Raw output logits, shape [B, C, N].
                - preds: Predicted labels, shape [B, N].
        """
        logits = self.model(points)
        loss = torch.nn.functional.nll_loss(logits, pc_labels)
        preds = logits.argmax(dim=1)

        return loss, logits, preds
